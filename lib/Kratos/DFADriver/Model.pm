package Kratos::DFADriver::Model;

use strict;
use warnings;
use 5.020;

use parent 'Class::Accessor';

use Carp;
use Carp::Assert::More;
use List::Util qw(first uniq);
use XML::LibXML;

Kratos::DFADriver::Model->mk_ro_accessors(qw(class_name xml));

our $VERSION = '0.00';

sub new {
	my ( $class, %opt ) = @_;

	my $self = \%opt;

	$self->{parameter}   = {};
	$self->{states}      = {};
	$self->{transitions} = [];
	$self->{xml} = XML::LibXML->load_xml( location => $self->{xml_file} );

	bless( $self, $class );

	$self->parse_xml;

	return $self;
}

sub new_from_repo {
	my ( $class, %opt ) = @_;
	my $repo       = $opt{repo};
	my $class_name = $opt{class_name};

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
			else {
				say "wat $attrib";
			}
		}
	}

	@states = uniq @states;
	@states = sort @states;

	my $xml = XML::LibXML::Document->new('1.0');

	my $data_node        = $xml->createElement('data');
	my $driver_node      = $xml->createElement('driver');
	my $states_node      = $xml->createElement('states');
	my $transitions_node = $xml->createElement('transitions');

	$driver_node->setAttribute( name => $class_name );

	$xml->setDocumentElement($data_node);
	$data_node->appendChild($driver_node);
	$driver_node->appendChild($states_node);
	$driver_node->appendChild($transitions_node);

	for my $state (@states) {
		my $state_node = $xml->createElement('state');
		$state_node->setAttribute( name => $state );
		$states_node->appendChild($state_node);
	}
	for my $transition ( sort keys %transition ) {
		my $transition_node = $xml->createElement('transition');
		my $level = $transition eq 'epilogue' ? 'epilogue' : 'user';
		$transition_node->setAttribute( name => $transition );
		for my $src_state ( @{ $transition{$transition}{src} } ) {
			my $src_node = $xml->createElement('src');
			my $src_name = $xml->createTextNode($src_state);
			$src_node->appendChild($src_name);
			$transition_node->appendChild($src_node);
		}
		for my $dst_state ( @{ $transition{$transition}{dst} } ) {
			my $dst_node = $xml->createElement('dst');
			my $dst_name = $xml->createTextNode($dst_state);
			$dst_node->appendChild($dst_name);
			$transition_node->appendChild($dst_node);
		}
		my $level_node = $xml->createElement('level');
		my $level_name = $xml->createTextNode($level);
		$level_node->appendChild($level_name);
		$transition_node->appendChild($level_node);
		$transitions_node->appendChild($transition_node);
	}

	$xml->toFile( $opt{xml_file} );

	return $class->new(%opt);
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

		$ret->{function}{$name}{raw}  = $function;
		$ret->{function}{$name}{node} = $function_node;

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
	my $transition_index = 0;

	for my $state_node ( $xml->findnodes('/data/driver/states/state') ) {
		my $name = $state_node->getAttribute('name');
		my $power = $state_node->getAttribute('power') // 0;
		$self->{states}{$name} = {
			power => $self->parse_xml_property( $state_node, 'power' ),
			id    => $state_index,
			node  => $state_node,
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
			id          => $transition_index,
			affects     => {%affects},
			node        => $transition_node,
		};

		push( @{ $self->{transitions} }, $transition );

		$transition_index++;
	}

	if ( my ($node) = $xml->findnodes('/data/startup/code') ) {
		$self->{startup}{code} = $node->textContent;
	}
	if ( my ($node) = $xml->findnodes('/data/heap/code') ) {
		$self->{heap}{code} = $node->textContent;
	}
	if ( my ($node) = $xml->findnodes('/data/after-transition/code') ) {
		$self->{after_transition}{code} = $node->textContent;
	}
	for my $node ( $xml->findnodes('/data/after-transition/if') ) {
		my $state = $node->getAttribute('state');
		for my $transition ( $node->findnodes('./transition') ) {
			my $name = $transition->getAttribute('name');
			push( @{ $self->{after_transition}{in_state}{$state} }, $name );
		}
	}
	if ( my ($node) = $xml->findnodes('/data/shutdown/code') ) {
		$self->{shutdown}{code} = $node->textContent;
	}

	return $self;
}

sub reset_property {
	my ( $self, $node, $name ) = @_;

	my ($property_node) = $node->findnodes("./${name}");

	if ($property_node) {
		for my $attr_node ( $property_node->findnodes('./static | ./lut') ) {
			$property_node->removeChild($attr_node);
		}
		for my $function_parent ( $property_node->findnodes('./function') ) {
			for my $function_node ( $function_parent->childNodes ) {
				if (   $function_node->nodeName eq 'user'
					or $function_node->nodeName eq 'user_arg' )
				{
					for my $attrnode ( $function_node->attributes ) {
						$attrnode->setValue(1);
					}
				}
				else {
					$function_parent->removeChild($function_node);
				}
			}
		}
	}
}

sub reset {
	my ($self) = @_;

	for my $state ( values %{ $self->{states} } ) {
		for my $property (qw(power)) {
			$self->reset_property( $state->{node}, $property );
		}
	}

	for my $transition ( @{ $self->{transitions} } ) {
		for my $property (
			qw(duration energy rel_energy_prev rel_energy_next timeout))
		{
			$self->reset_property( $transition->{node}, $property );
		}
	}
}

sub set_state_power {
	my ( $self, $state, $power ) = @_;
	my $state_node = $self->{states}{$state}{node};

	$power = sprintf( '%.f', $power );

	printf( "state %-16s: adjust power %d -> %d µW\n",
		$state, $self->{states}{$state}{power}{static}, $power );

	$self->{states}{$state}{power}{static} = $power;

	my ($static_parent) = $state_node->findnodes('./power');
	if ( not $static_parent ) {
		$static_parent = XML::LibXML::Element->new('power');
		$state_node->appendChild($static_parent);
	}

	for my $static_node ( $static_parent->findnodes('./static') ) {
		$static_parent->removeChild($static_node);
	}

	my $static_node = XML::LibXML::Element->new('static');
	my $text_node   = XML::LibXML::Text->new($power);

	$text_node->setData($power);
	$static_node->appendChild($text_node);
	$static_parent->appendChild($static_node);
}

sub set_transition_property {
	my ( $self, $transition_name, $property, $value ) = @_;

	if ( not defined $value ) {
		return;
	}

	my $transition      = $self->get_transition_by_name($transition_name);
	my $transition_node = $transition->{node};
	$value = sprintf( '%.f', $value );

	printf( "transition %-16s: adjust %s %d -> %d\n",
		$transition->{name}, $property, $transition->{$property}{static},
		$value );

	$transition->{$property}{static} = $value;

	my ($static_parent) = $transition_node->findnodes("./${property}");
	if ( not $static_parent ) {
		$static_parent = XML::LibXML::Element->new($property);
		$transition_node->appendChild($static_parent);
	}

	for my $static_node ( $static_parent->findnodes('./static') ) {
		$static_parent->removeChild($static_node);
	}

	my $static_node = XML::LibXML::Element->new('static');
	my $text_node   = XML::LibXML::Text->new($value);

	$text_node->setData($value);
	$static_node->appendChild($text_node);
	$static_parent->appendChild($static_node);
}

sub set_state_lut {
	my ( $self, $state, $property, $lut ) = @_;
	my $state_node = $self->{states}{$state}{node};

	if ( not defined $lut ) {
		return;
	}

	my ($lut_parent) = $state_node->findnodes("./${property}");
	for my $lut_node ( $lut_parent->findnodes('./lut') ) {
		$lut_parent->removeChild($lut_node);
	}

	my $lut_node = XML::LibXML::Element->new('lut');
	$lut_parent->appendChild($lut_node);

	for my $lut_entry ( @{$lut} ) {
		my $entry_node = XML::LibXML::Element->new('entry');
		my $value_node = XML::LibXML::Text->new( $lut_entry->{value} );
		for my $param ( sort keys %{ $lut_entry->{key} } ) {
			$entry_node->setAttribute( $param, $lut_entry->{key}{$param} );
		}
		$entry_node->appendChild($value_node);
		$lut_node->appendChild($entry_node);
	}
}

sub set_transition_lut {
	my ( $self, $transition_name, $property, $lut ) = @_;

	if ( not defined $lut ) {
		return;
	}

	my $transition      = $self->get_transition_by_name($transition_name);
	my $transition_node = $transition->{node};

	my ($lut_parent) = $transition_node->findnodes("./${property}");
	for my $lut_node ( $lut_parent->findnodes('./lut') ) {
		$lut_parent->removeChild($lut_node);
	}

	my $lut_node = XML::LibXML::Element->new('lut');
	$lut_parent->appendChild($lut_node);

	for my $lut_entry ( @{$lut} ) {
		my $entry_node = XML::LibXML::Element->new('entry');
		my $value_node = XML::LibXML::Text->new( $lut_entry->{value} );
		for my $param ( sort keys %{ $lut_entry->{key} } ) {
			$entry_node->setAttribute( $param, $lut_entry->{key}{$param} );
		}
		$entry_node->appendChild($value_node);
		$lut_node->appendChild($entry_node);
	}
}

sub set_state_params {
	my ( $self, $state, $fun_name, $function, @params ) = @_;
	my $old_params = 'None';
	my $state_node = $self->{states}{$state}{node};

	if ( exists $self->{states}{$state}{power}{function}{$fun_name} ) {
		$old_params = join( q{ },
			@{ $self->{states}{$state}{power}{function}{$fun_name}{params} } );
	}

	printf( "state %-16s: adjust %s power function parameters [%s] -> [%s]\n",
		$state, $fun_name, $old_params, join( q{ }, @params ) );

	my ($function_parent) = $state_node->findnodes('./power/function');

	if ( not $function_parent ) {
		my ($power_node) = $state_node->findnodes('./power');
		$function_parent = XML::LibXML::Element->new('function');
		$power_node->appendChild($function_parent);
	}

	for my $function_node ( $function_parent->findnodes("./${fun_name}") ) {
		$function_parent->removeChild($function_node);
	}

	my $function_node    = XML::LibXML::Element->new($fun_name);
	my $function_content = XML::LibXML::CDATASection->new($function);

	$function_node->appendChild($function_content);
	$function_parent->appendChild($function_node);

	for my $i ( 0 .. $#params ) {
		$self->{states}{$state}{power}{function}{$fun_name}{params}[$i]
		  = $params[$i];
		$function_node->setAttribute( "param$i", $params[$i] );
	}
}

sub set_transition_params {
	my ( $self, $transition_name, $fun_type, $fun_name, $function, @params )
	  = @_;
	my $transition      = $self->get_transition_by_name($transition_name);
	my $transition_node = $transition->{node};
	my $old_params      = 'None';

	if ( exists $transition->{$fun_type}{function}{$fun_name} ) {
		$old_params = join( q{ },
			@{ $transition->{$fun_type}{function}{$fun_name}{params} } );
	}

	printf( "transition %-16s: adjust %s %s function parameters [%s] -> [%s]\n",
		$transition_name, $fun_name, $fun_type, $old_params,
		join( q{ }, @params ) );

	my ($function_parent)
	  = $transition_node->findnodes("./${fun_type}/function");

	if ( not $function_parent ) {
		my ($property_node) = $transition_node->findnodes("./${fun_type}");
		$function_parent = XML::LibXML::Element->new('function');
		$property_node->appendChild($function_parent);
	}

	for my $function_node ( $function_parent->findnodes("./${fun_name}") ) {
		$function_parent->removeChild($function_node);
	}

	my $function_node    = XML::LibXML::Element->new($fun_name);
	my $function_content = XML::LibXML::CDATASection->new($function);

	$function_node->appendChild($function_content);
	$function_parent->appendChild($function_node);

	for my $i ( 0 .. $#params ) {
		$transition->{$fun_type}{function}{$fun_name}{params}[$i] = $params[$i];
		$function_node->setAttribute( "param$i", $params[$i] );
	}
}

sub set_voltage {
	my ( $self, $min_voltage, $max_voltage ) = @_;

	my ($data_node) = $self->xml->findnodes('/data');

	for my $voltage_node ( $data_node->findnodes('./voltage') ) {
		$data_node->removeChild($voltage_node);
	}

	my $voltage_node = XML::LibXML::Element->new('voltage');
	$voltage_node->setAttribute( 'min', $min_voltage );
	$voltage_node->setAttribute( 'max', $max_voltage );

	$data_node->appendChild($voltage_node);
}

sub save {
	my ($self) = @_;

	$self->{xml}->toFile( $self->{xml_file} );
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

	return $self->{startup}{code} // q{};
}

sub heap_code {
	my ($self) = @_;

	return $self->{heap}{code} // q{};
}

sub after_transition_code {
	my ($self) = @_;

	return $self->{after_transition}{code} // q{};
}

sub get_state_extra_transitions {
	my ( $self, $state ) = @_;

	return @{ $self->{after_transition}{in_state}{$state} // [] };
}

sub shutdown_code {
	my ($self) = @_;

	return $self->{shutdown}{code} // q{};
}

sub get_transition_by_name {
	my ( $self, $name ) = @_;

	my $transition = first { $_->{name} eq $name } @{ $self->{transitions} };

	return $transition;
}

sub get_transition_by_id {
	my ( $self, $id ) = @_;

	return $self->{transitions}[$id];
}

sub get_state_id {
	my ( $self, $name ) = @_;

	return $self->{states}{$name}{id};
}

sub get_state_name {
	my ( $self, $id ) = @_;

	return ( $self->get_state_enum )[$id];
}

sub get_state_power {
	my ( $self, $name ) = @_;

	return $self->{states}{$name}{power}{static};
}

sub get_state_power_with_params {
	my ( $self, $name, $param_values ) = @_;

	my $hash_str = join( ';',
		map  { $param_values->{$_} }
		sort { $a cmp $b } keys %{$param_values} );

	if ( $hash_str eq q{} ) {
		return $self->get_state_power($name);
	}

	if ( exists $self->{states}{$name}{power}{lut}{$hash_str} ) {
		return $self->{states}{$name}{power}{lut}{$hash_str};
	}

	say "Note: No matching LUT for state ${name}, using median";

	return $self->get_state_power($name);
}

sub get_state_enum {
	my ($self) = @_;

	if ( not exists $self->{state_enum} ) {
		@{ $self->{state_enum} }
		  = sort { $self->{states}{$a}{id} <=> $self->{states}{$b}{id} }
		  keys %{ $self->{states} };
	}

	return @{ $self->{state_enum} };
}

sub transitions {
	my ($self) = @_;

	return @{ $self->{transitions} };
}

sub TO_JSON {
	my ($self) = @_;

	my %state_copy
	  = map { $_ => { %{ $self->{states}{$_} } } } keys %{ $self->{states} };
	my %transition_copy
	  = map { $_->{name} => { %{$_} } } @{ $self->{transitions} };

	for my $val ( values %state_copy ) {
		delete $val->{node};
		if ( exists $val->{power}{function} ) {
			$val->{power} = { %{ $val->{power} } };
			$val->{power}{function} = { %{ $val->{power}{function} } };
			for my $key ( keys %{ $val->{power}{function} } ) {
				$val->{power}{function}{$key}
				  = { %{ $val->{power}{function}{$key} } };
				delete $val->{power}{function}{$key}{node};
			}
		}
	}
	for my $val ( values %transition_copy ) {
		delete $val->{node};
		for
		  my $key (qw(duration energy rel_energy_prev rel_energy_next timeout))
		{
			if ( exists $val->{$key}{function} ) {
				$val->{$key} = { %{ $val->{$key} } };
				$val->{$key}{function} = { %{ $val->{$key}{function} } };
				for my $ftype ( keys %{ $val->{$key}{function} } ) {
					$val->{$key}{function}{$ftype}
					  = { %{ $val->{$key}{function}{$ftype} } };
					delete $val->{$key}{function}{$ftype}{node};
				}
			}
		}
	}

	my $json = {
		parameter  => $self->{parameter},
		state      => {%state_copy},
		transition => {%transition_copy},
	};

	return $json;
}

1;
