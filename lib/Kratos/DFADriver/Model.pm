package Kratos::DFADriver::Model;

use strict;
use warnings;
use 5.020;

use parent 'Class::Accessor';

use Carp;
use Carp::Assert::More;
use List::Util qw(first uniq);
use File::Slurp qw(read_file write_file);
use JSON;
use XML::LibXML;

Kratos::DFADriver::Model->mk_ro_accessors(
	qw(class_name parameter state transition));

our $VERSION = '0.00';

sub new {
	my ( $class, %opt ) = @_;

	my $self = \%opt;

	$self->{custom_code} = {};
	$self->{parameter}   = {};
	$self->{state}       = {};
	$self->{transition}  = {};
	$self->{voltage}     = {};

	bless( $self, $class );

	if ( $self->{model_file} =~ m{ [.] xml $ }x ) {
		$self->{xml} = XML::LibXML->load_xml( location => $self->{model_file} );
		$self->parse_xml;
		$self->{model_file} =~ s{ [.] xml $}{.json}x;
		write_file( $self->{model_file},
			JSON->new->pretty->encode( $self->TO_JSON ) );
	}
	else {
		my $json = JSON->new->decode( scalar read_file( $self->{model_file} ) );
		for my $key (qw(custom_code parameter state transition)) {
			$self->{$key} = $json->{$key};
		}
		$self->{class_name} = $json->{class};
	}

	return $self;
}

sub new_from_repo {
	my ( $class, %opt ) = @_;
	my $repo = $opt{repo};

	my $self = {
		class_name => $opt{class_name},
		model_file => $opt{model_file},
		voltage    => {},
	};

	bless( $self, $class );

	my $class_name = $self->{class_name};

	my @states;
	my %transition;

	if ( not exists $repo->{class}{$class_name} ) {
		die("Unknown class: $class_name\n");
	}
	my $class_base = $repo->{class}{$class_name};

	for my $function ( values %{ $class_base->{function} } ) {
		for my $attrib ( @{ $function->{attributes} // [] } ) {
			if ( $attrib =~ s{ ^ src _ }{}x ) {
				push( @states,                                    $attrib );
				push( @{ $transition{ $function->{name} }{src} }, $attrib );
			}
			elsif ( $attrib =~ s{ ^ dst _ }{}x ) {
				push( @states,                                    $attrib );
				push( @{ $transition{ $function->{name} }{dst} }, $attrib );
			}
			elsif ( $attrib =~ m{ ^ epilogue $ }x ) {
				$transition{ $function->{name} }{level} = 'epilogue';
			}
			else {
				say "wat $attrib";
			}
		}
	}

	@states = uniq @states;
	@states = sort @states;

	for my $i ( 0 .. $#states ) {
		$self->{state}{ $states[$i] } = {
			id    => $i,
			power => {
				static => 0,
			}
		};
	}

	my @transition_names = sort keys %transition;

	for my $i ( 0 .. $#transition_names ) {
		my $name = $transition_names[$i];
		my $guess_level = ( $name eq 'epilogue' ? 'epilogue' : 'user' );
		$self->{transition}{$name} = {
			name        => $name,
			id          => $i,
			destination => $transition{$name}{dst}[0],
			origins     => $transition{$name}{src},
			level       => $transition{$name}{level} // $guess_level,
		};
	}

	write_file( $self->{model_file},
		JSON->new->pretty->encode( $self->TO_JSON ) );

	return $self;
}

sub parse_xml_property {
	my ( $self, $node, $property_name ) = @_;

	my $xml = $self->{xml};
	my $ret = { static => 0 };

	my ($property_node) = $node->findnodes("./${property_name}");
	if ( not $property_node ) {
		return $ret;
	}

	for my $static_node ( $property_node->findnodes('./static') ) {
		$ret->{static} = 0 + $static_node->textContent;
	}
	for my $function_node ( $property_node->findnodes('./function/*') ) {
		my $name     = $function_node->nodeName;
		my $function = $function_node->textContent;
		$function =~ s{^ \n* \s* }{}x;
		$function =~ s{\s* \n* $}{}x;
		$function =~ s{ [\n\t]+ }{}gx;

		$ret->{function}{$name}{raw} = $function;

		my $param_idx = 0;
		while ( $function_node->hasAttribute("param${param_idx}") ) {
			push(
				@{ $ret->{function}{$name}{params} },
				$function_node->getAttribute("param${param_idx}")
			);
			$param_idx++;
		}
	}
	for my $lut_node ( $property_node->findnodes('./lut/*') ) {
		my @paramkey = map { $_->[0]->getValue }
		  sort { $a->[1] cmp $b->[1] }
		  map { [ $_, $_->nodeName ] } @{ $lut_node->attributes->nodes };
		$ret->{lut}{ join( ';', @paramkey ) } = 0 + $lut_node->textContent;
	}

	return $ret;
}

sub parse_xml {
	my ($self) = @_;

	my $xml = $self->{xml};
	my ($driver_node) = $xml->findnodes('/data/driver');
	my $class_name  = $self->{class_name} = $driver_node->getAttribute('name');
	my $state_index = 0;
	my @transitions;

	for my $state_node ( $xml->findnodes('/data/driver/states/state') ) {
		my $name = $state_node->getAttribute('name');
		my $power = $state_node->getAttribute('power') // 0;
		$self->{state}{$name} = {
			power => $self->parse_xml_property( $state_node, 'power' ),
			id    => $state_index,
		};

		$state_index++;
	}

	for my $param_node ( $xml->findnodes('/data/driver/parameters/param') ) {
		my $param_name    = $param_node->getAttribute('name');
		my $function_name = $param_node->getAttribute('functionname');
		my $function_arg  = $param_node->getAttribute('functionparam');

		$self->{parameter}{$param_name} = {
			function => $function_name,
			arg_name => $function_arg,
			default  => undef,
		};
	}

	for my $transition_node (
		$xml->findnodes('/data/driver/transitions/transition') )
	{
		my @src_nodes      = $transition_node->findnodes('./src');
		my ($dst_node)     = $transition_node->findnodes('./dst');
		my ($level_node)   = $transition_node->findnodes('./level');
		my @param_nodes    = $transition_node->findnodes('./param');
		my @affected_nodes = $transition_node->findnodes('./affects/param');
		my @parameters;
		my %affects;

		my @source_states = map { $_->textContent } @src_nodes;

		for my $param_node (@param_nodes) {
			my @value_nodes = $param_node->findnodes('./value');
			my $param       = {
				name   => $param_node->getAttribute('name'),
				values => [ map { $_->textContent } @value_nodes ],
			};
			push( @parameters, $param );
		}

		for my $param_node (@affected_nodes) {
			my $param_name  = $param_node->getAttribute('name');
			my $param_value = $param_node->getAttribute('value');
			$affects{$param_name} = $param_value;
		}

		my $transition = {
			name => $transition_node->getAttribute('name'),
			duration =>
			  $self->parse_xml_property( $transition_node, 'duration' ),
			energy => $self->parse_xml_property( $transition_node, 'energy' ),
			rel_energy_prev =>
			  $self->parse_xml_property( $transition_node, 'rel_energy_prev' ),
			rel_energy_next =>
			  $self->parse_xml_property( $transition_node, 'rel_energy_next' ),
			timeout => $self->parse_xml_property( $transition_node, 'timeout' ),
			parameters  => [@parameters],
			origins     => [@source_states],
			destination => $dst_node->textContent,
			level       => $level_node->textContent,
			affects     => {%affects},
		};

		push( @transitions, $transition );
	}

	@transitions = sort { $a->{name} cmp $b->{name} } @transitions;
	for my $i ( 0 .. $#transitions ) {
		$transitions[$i]{id} = $i;
		$self->{transition}{ $transitions[$i]{name} } = $transitions[$i];
	}

	if ( my ($node) = $xml->findnodes('/data/startup/code') ) {
		$self->{custom_code}{startup} = $node->textContent;
	}
	if ( my ($node) = $xml->findnodes('/data/heap/code') ) {
		$self->{custom_code}{heap} = $node->textContent;
	}
	if ( my ($node) = $xml->findnodes('/data/after-transition/code') ) {
		$self->{custom_code}{after_transition} = $node->textContent;
	}
	for my $node ( $xml->findnodes('/data/after-transition/if') ) {
		my $state = $node->getAttribute('state');
		for my $transition ( $node->findnodes('./transition') ) {
			my $name = $transition->getAttribute('name');
			push(
				@{ $self->{custom_code}{after_transition_by_state}{$state} },
				$name
			);
		}
	}
	if ( my ($node) = $xml->findnodes('/data/shutdown/code') ) {
		$self->{custom_code}{shutdown} = $node->textContent;
	}

	return $self;
}

sub reset_property {
	my ( $self, $hash, $name ) = @_;

	delete $hash->{$name}{static};
	if ( exists $hash->{$name}{function} ) {
		delete $hash->{$name}{function}{estimate};
	}
	if ( exists $hash->{$name}{function}{user} ) {
		$hash->{$name}{function}{user}{params}
		  = [ map { 1 } @{ $hash->{$name}{function}{user}{params} } ];
	}
}

sub reset {
	my ($self) = @_;

	for my $state ( values %{ $self->{state} } ) {
		for my $property (qw(power)) {
			$self->reset_property( $state, $property );
		}
	}

	for my $transition ( $self->transitions ) {
		for my $property (
			qw(duration energy rel_energy_prev rel_energy_next timeout))
		{
			$self->reset_property( $transition, $property );
		}
	}
}

sub set_state_power {
	my ( $self, $state, $power ) = @_;

	$power = sprintf( '%.f', $power );

	printf( "state %-16s: adjust power %d -> %d µW\n",
		$state, $self->{state}{$state}{power}{static}, $power );

	$self->{state}{$state}{power}{static} = $power;
}

sub set_transition_property {
	my ( $self, $transition_name, $property, $value ) = @_;

	if ( not defined $value ) {
		return;
	}

	my $transition = $self->get_transition_by_name($transition_name);

	$value = sprintf( '%.f', $value );

	printf( "transition %-16s: adjust %s %d -> %d\n",
		$transition->{name}, $property, $transition->{$property}{static},
		$value );

	$transition->{$property}{static} = $value;
}

sub set_state_lut {
	my ( $self, $state, $property, $lut ) = @_;

	if ( not defined $lut ) {
		return;
	}

	...;
}

sub set_transition_lut {
	my ( $self, $transition_name, $property, $lut ) = @_;

	if ( not defined $lut ) {
		return;
	}

	...;
}

sub set_state_params {
	my ( $self, $state, $fun_name, $function, @params ) = @_;
	my $old_params = 'None';

	if ( exists $self->{state}{$state}{power}{function}{$fun_name} ) {
		$old_params = join( q{ },
			@{ $self->{state}{$state}{power}{function}{$fun_name}{params} } );
	}

	printf( "state %-16s: adjust %s power function parameters [%s] -> [%s]\n",
		$state, $fun_name, $old_params, join( q{ }, @params ) );

	$self->{state}{$state}{power}{function}{$fun_name}{raw} = $function;
	for my $i ( 0 .. $#params ) {
		$self->{state}{$state}{power}{function}{$fun_name}{params}[$i]
		  = $params[$i];
	}
}

sub set_transition_params {
	my ( $self, $transition_name, $fun_type, $fun_name, $function, @params )
	  = @_;
	my $transition = $self->get_transition_by_name($transition_name);
	my $old_params = 'None';

	if ( exists $transition->{$fun_type}{function}{$fun_name} ) {
		$old_params = join( q{ },
			@{ $transition->{$fun_type}{function}{$fun_name}{params} } );
	}

	printf( "transition %-16s: adjust %s %s function parameters [%s] -> [%s]\n",
		$transition_name, $fun_name, $fun_type, $old_params,
		join( q{ }, @params ) );

	$transition->{$fun_type}{function}{$fun_name}{raw} = $function;
	for my $i ( 0 .. $#params ) {
		$transition->{$fun_type}{function}{$fun_name}{params}[$i] = $params[$i];
	}
}

sub set_voltage {
	my ( $self, $min_voltage, $max_voltage ) = @_;

	$self->{voltage} = {
		min => $min_voltage,
		max => $max_voltage,
	};
}

sub save {
	my ($self) = @_;

	write_file( $self->{model_file},
		JSON->new->pretty->encode( $self->TO_JSON ) );
}

sub parameter_hash {
	my ($self) = @_;

	for my $param_name ( keys %{ $self->{parameter} } ) {
		$self->{parameter}{$param_name}{value}
		  = $self->{parameter}{$param_name}{default};
	}

	return %{ $self->{parameter} };
}

sub update_parameter_hash {
	my ( $self, $param_hash, $function, @args ) = @_;

	my $transition = $self->get_transition_by_name($function);

	for my $param ( keys %{ $transition->{affects} } ) {
		$param_hash->{$param}{value} = $transition->{affects}{$param};
	}

	for my $i ( 0 .. $#args ) {
		my $arg_name  = $transition->{parameters}[$i]{name};
		my $arg_value = $args[$i];

		for my $param_name ( keys %{ $self->{parameter} } ) {
			if ( $self->{parameter}{$param_name}{arg_name} eq $arg_name ) {
				$param_hash->{$param_name}{value} = $arg_value;
			}
		}
	}
}

sub startup_code {
	my ($self) = @_;

	return $self->{custom_code}{startup} // q{};
}

sub heap_code {
	my ($self) = @_;

	return $self->{custom_code}{heap} // q{};
}

sub after_transition_code {
	my ($self) = @_;

	return $self->{custom_code}{after_transition} // q{};
}

sub get_state_extra_transitions {
	my ( $self, $state ) = @_;

	return @{ $self->{custom_code}{after_transition_by_state}{$state} // [] };
}

sub shutdown_code {
	my ($self) = @_;

	return $self->{custom_code}{shutdown} // q{};
}

sub get_transition_by_name {
	my ( $self, $name ) = @_;

	return $self->{transition}{$name};
}

sub get_transition_by_id {
	my ( $self, $id ) = @_;

	my $transition = first { $_->{id} == $id } $self->transitions;

	return $transition;
}

sub get_state_id {
	my ( $self, $name ) = @_;

	return $self->{state}{$name}{id};
}

sub get_state_name {
	my ( $self, $id ) = @_;

	return ( $self->get_state_enum )[$id];
}

sub get_state_power {
	my ( $self, $name ) = @_;

	return $self->{state}{$name}{power}{static};
}

sub get_state_power_with_params {
	my ( $self, $name, $param_values ) = @_;

	my $hash_str = join( ';',
		map  { $param_values->{$_} }
		sort { $a cmp $b } keys %{$param_values} );

	if ( $hash_str eq q{} ) {
		return $self->get_state_power($name);
	}

	if ( exists $self->{state}{$name}{power}{lut}{$hash_str} ) {
		return $self->{state}{$name}{power}{lut}{$hash_str};
	}

	say "Note: No matching LUT for state ${name}, using median";

	return $self->get_state_power($name);
}

sub get_state_enum {
	my ($self) = @_;

	if ( not exists $self->{state_enum} ) {
		@{ $self->{state_enum} }
		  = sort { $self->{state}{$a}{id} <=> $self->{state}{$b}{id} }
		  keys %{ $self->{state} };
	}

	return @{ $self->{state_enum} };
}

sub transitions {
	my ($self) = @_;

	my @ret = values %{ $self->{transition} };
	@ret = sort { $a->{id} <=> $b->{id} } @ret;
	return @ret;
}

sub TO_JSON {
	my ($self) = @_;

	return {
		class       => $self->{class_name},
		parameter   => $self->{parameter},
		state       => $self->{state},
		transition  => $self->{transition},
		custom_code => $self->{custom_code},
		voltage     => $self->{voltage},
	};
}

1;
