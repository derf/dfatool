package Kratos::DFADriver::Model;

use strict;
use warnings;
use 5.020;

use parent 'Class::Accessor';

use Carp;
use Carp::Assert::More;
use List::Util qw(first);
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
			power => { static => 0+$power },
			id    => $state_index,
			node  => $state_node,
		};

		for my $fun_node ( $state_node->findnodes('./powerfunction/*') ) {
			my $fname         = $fun_node->nodeName;
			my $powerfunction = $fun_node->textContent;
			$powerfunction =~ s{^ \n* \s* }{}x;
			$powerfunction =~ s{\s* \n* $}{}x;
			$powerfunction =~ s{ [\n\t]+ }{}gx;
			$self->{states}{$name}{power}{function}{$fname}{raw}
			  = $powerfunction;
			$self->{states}{$name}{power}{function}{$fname}{node} = $fun_node;
			my $attrindex = 0;

			while ( $fun_node->hasAttribute("param${attrindex}") ) {
				push(
					@{
						$self->{states}{$name}{power}{function}{$fname}{params}
					},
					$fun_node->getAttribute("param${attrindex}")
				);
				$attrindex++;
			}
		}

		$state_index++;
	}

	for my $param_node ( $xml->findnodes('/data/driver/parameters/param') ) {
		my $param_name    = $param_node->getAttribute('name');
		my $function_name = $param_node->getAttribute('functionname');
		my $function_arg  = $param_node->getAttribute('functionparam');
		my $default       = $param_node->textContent;

		$self->{parameter}{$param_name} = {
			function => $function_name,
			arg_name => $function_arg,
			default  => $default,
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
			name        => $transition_node->getAttribute('name'),
			duration    => { static => 0+($transition_node->getAttribute('duration') // 0) },
			energy      => { static => 0+($transition_node->getAttribute('energy') // 0) },
			rel_energy  => { static => 0+($transition_node->getAttribute('rel_energy') // 0) },
			parameters  => [@parameters],
			origins     => [@source_states],
			destination => $dst_node->textContent,
			level       => $level_node->textContent,
			id          => $transition_index,
			affects     => {%affects},
			node        => $transition_node,
		};

		for my $fun_node ( $transition_node->findnodes('./timeoutfunction/*') )
		{
			my $name     = $fun_node->nodeName;
			my $function = $fun_node->textContent;
			$function =~ s{^ \n* \s* }{}x;
			$function =~ s{\s* \n* $}{}x;
			$transition->{timeout}{function}{$name}{raw}  = $function;
			$transition->{timeout}{function}{$name}{node} = $fun_node;
			my $attrindex = 0;
			while ( $fun_node->hasAttribute("param${attrindex}") ) {
				push(
					@{ $transition->{timeout}{function}{$name}{params} },
					$fun_node->getAttribute("param${attrindex}")
				);
				$attrindex++;
			}
		}

		for my $fun_node ( $transition_node->findnodes('./durationfunction/*') )
		{
			my $name     = $fun_node->nodeName;
			my $function = $fun_node->textContent;
			$function =~ s{^ \n* \s* }{}x;
			$function =~ s{\s* \n* $}{}x;
			$transition->{duration}{function}{$name}{raw}  = $function;
			$transition->{duration}{function}{$name}{node} = $fun_node;
			my $attrindex = 0;
			while ( $fun_node->hasAttribute("param${attrindex}") ) {
				push(
					@{ $transition->{duration}{function}{$name}{params} },
					$fun_node->getAttribute("param${attrindex}")
				);
				$attrindex++;
			}
		}

		for my $fun_node ( $transition_node->findnodes('./energyfunction/*') )
		{
			my $name     = $fun_node->nodeName;
			my $function = $fun_node->textContent;
			$function =~ s{^ \n* \s* }{}x;
			$function =~ s{\s* \n* $}{}x;
			$transition->{energy}{function}{$name}{raw}  = $function;
			$transition->{energy}{function}{$name}{node} = $fun_node;
			my $attrindex = 0;
			while ( $fun_node->hasAttribute("param${attrindex}") ) {
				push(
					@{ $transition->{energy}{function}{$name}{params} },
					$fun_node->getAttribute("param${attrindex}")
				);
				$attrindex++;
			}
		}

		for my $fun_node ( $transition_node->findnodes('./rel_energyfunction/*') )
		{
			my $name     = $fun_node->nodeName;
			my $function = $fun_node->textContent;
			$function =~ s{^ \n* \s* }{}x;
			$function =~ s{\s* \n* $}{}x;
			$transition->{rel_energy}{function}{$name}{raw}  = $function;
			$transition->{rel_energy}{function}{$name}{node} = $fun_node;
			my $attrindex = 0;
			while ( $fun_node->hasAttribute("param${attrindex}") ) {
				push(
					@{ $transition->{rel_energy}{function}{$name}{params} },
					$fun_node->getAttribute("param${attrindex}")
				);
				$attrindex++;
			}
		}

		push( @{ $self->{transitions} }, $transition );

		$transition_index++;
	}

	if ( my ($node) = $xml->findnodes('/data/startup/code') ) {
		$self->{startup}{code} = $node->textContent;
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

sub reset {
	my ($self) = @_;

	for my $state (values %{$self->{states}}) {
		$state->{node}->removeAttribute('power');
		for my $list_node (@{$state->{node}->findnodes('./powerfunction')}) {
			for my $fun_name (keys %{$state->{power}{function}}) {
				my $fun_node = $state->{power}{function}{$fun_name}{node};
				if ($fun_node->nodeName eq 'user') {
					for my $attrnode ($fun_node->attributes) {
						$attrnode->setValue(1);
					}
				}
				else {
					$list_node->removeChild($fun_node);
				}
			}
		}
	}
	for my $transition (@{$self->{transitions}}) {
		$transition->{node}->removeAttribute('duration');
		$transition->{node}->removeAttribute('energy');
		$transition->{node}->removeAttribute('rel_energy');
		for my $list_node (@{$transition->{node}->findnodes('./timeoutfunction')}) {
			for my $fun_name (keys %{$transition->{timeout}{function}}) {
				my $fun_node = $transition->{timeout}{function}{$fun_name}{node};
				if ($fun_node->nodeName eq 'user') {
					for my $attrnode ($fun_node->attributes) {
						$attrnode->setValue(1);
					}
				}
				else {
					$list_node->removeChild($fun_node);
				}
			}
		}
	}
}

sub set_state_power {
	my ( $self, $state, $power ) = @_;

	$power = sprintf( '%.f', $power );

	printf( "state %-16s: adjust power %d -> %d µW\n",
		$state, $self->{states}{$state}{power}{static}, $power );

	$self->{states}{$state}{power}{static} = $power;
	$self->{states}{$state}{node}->setAttribute( 'power', $power );
}

sub set_state_params {
	my ( $self, $state, $fun_name, $function, @params ) = @_;
	my $old_params = 'None';

	if ( exists $self->{states}{$state}{power}{function}{$fun_name} ) {
		$old_params = join( q{ },
			@{ $self->{states}{$state}{power}{function}{$fun_name}{params} } );
	}

	printf( "state %-16s: adjust %s power function parameters [%s] -> [%s]\n",
		$state, $fun_name, $old_params, join( q{ }, @params ) );

	if ( not defined $self->{states}{$state}{power}{function}{$fun_name}{node} )
	{
		my ($fun_node)
		  = $self->{states}{$state}{node}->findnodes('./powerfunction');
		if ($fun_node) {
			my $new_node = XML::LibXML::Element->new($fun_name);
			$self->{states}{$state}{power}{function}{$fun_name}{node}
			  = $new_node;
			$fun_node->appendChild($new_node);
		}
		else {
			say
			  '  skipping XML write-back because of missing powerfunction node';
			return;
		}
	}

	if ( defined $function ) {
		my $cdata_node = XML::LibXML::CDATASection->new($function);
		$self->{states}{$state}{power}{function}{$fun_name}{node}
		  ->removeChildNodes;
		$self->{states}{$state}{power}{function}{$fun_name}{node}
		  ->appendChild($cdata_node);
	}

	for my $i ( 0 .. $#params ) {
		$self->{states}{$state}{power}{function}{$fun_name}{params}[$i]
		  = $params[$i];
		$self->{states}{$state}{power}{function}{$fun_name}{node}
		  ->setAttribute( "param$i", $params[$i] );
	}
}

sub set_transition_params {
	my ( $self, $transition_name, $fun_type, $fun_name, $function, @params ) = @_;
	my $transition = $self->get_transition_by_name($transition_name);
	my $old_params = 'None';

	if ( exists $transition->{$fun_type}{function}{$fun_name} ) {
		$old_params = join( q{ },
			@{ $transition->{$fun_type}{function}{$fun_name}{params} } );
	}

	printf(
		"transition %-16s: adjust %s %s function parameters [%s] -> [%s]\n",
		$transition_name, $fun_name, $fun_type, $old_params, join( q{ }, @params ) );

	if ( not defined $transition->{$fun_type}{function}{$fun_name}{node} ) {
		my ($fun_node) = $transition->{node}->findnodes("./${fun_type}function");
		if ($fun_node) {
			my $new_node = XML::LibXML::Element->new($fun_name);
			$transition->{$fun_type}{function}{$fun_name}{node} = $new_node;
			$fun_node->appendChild($new_node);
		}
		else {
			say
"  skipping XML write-back because of missing ${fun_type}function node";
			return;
		}
	}

	if ( defined $function ) {
		my $cdata_node = XML::LibXML::CDATASection->new($function);
		$transition->{$fun_type}{function}{$fun_name}{node}->removeChildNodes;
		$transition->{$fun_type}{function}{$fun_name}{node}
		  ->appendChild($cdata_node);
	}

	for my $i ( 0 .. $#params ) {
		$transition->{$fun_type}{function}{$fun_name}{params}[$i] = $params[$i];
		$transition->{$fun_type}{function}{$fun_name}{node}
		  ->setAttribute( "param$i", $params[$i] );
	}
}

sub set_transition_data {
	my ( $self, $transition_name, $duration, $energy, $rel_energy ) = @_;

	my $transition = $self->get_transition_by_name($transition_name);
	$duration = sprintf( '%.f', $duration );
	$energy = sprintf( '%.f', $energy );

	printf( 'transition %-16s: adjust duration %d -> %d µs',
		$transition->{name}, $transition->{duration}{static}, $duration);
	$transition->{duration}{static} = $duration;
	$transition->{node}->setAttribute('duration', $duration);

	printf( ', absolute energy %d -> %d pJ',
		$transition->{energy}{static}, $energy );

	$transition->{energy}{static} = $energy;
	$transition->{node}->setAttribute( 'energy', $energy );

	if (defined $rel_energy) {
		$rel_energy = sprintf('%.f', $rel_energy);
		printf( ", relative energy %d -> %d pJ\n",
			$transition->{rel_energy}{static}, $rel_energy );

		$transition->{rel_energy}{static} = $rel_energy;
		$transition->{node}->setAttribute( 'rel_energy', $rel_energy );
	}
	else {
		print("\n");
	}
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
		for my $key (qw(duration energy rel_energy timeout)) {
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
