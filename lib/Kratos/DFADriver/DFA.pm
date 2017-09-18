package Kratos::DFADriver::DFA;

use strict;
use warnings;
use 5.020;

use parent 'Class::Accessor';

use Data::Dumper;
use FLAT::DFA;
use Math::Cartesian::Product;

Kratos::DFADriver::DFA->mk_ro_accessors(qw(model));

our $VERSION = '0.00';

sub new {
	my ( $class, %opt ) = @_;

	my $self = \%opt;

	bless( $self, $class );

	return $self;
}

sub set_model {
	my ( $self, $model ) = @_;

	$self->{model} = $model;

	return $self;
}

sub reduced_id_to_state {
	my ( $self, $id ) = @_;

	if ( not( $self->{excluded_states} and @{ $self->{excluded_states} } ) ) {
		return $self->model->get_state_name($id);
	}

	my @excluded
	  = map { $self->model->get_state_id($_) } @{ $self->{excluded_states} };
	@excluded = reverse sort { $a <=> $b } @excluded;
	my @state_enum = $self->model->get_state_enum;

	for my $state (@excluded) {
		splice( @state_enum, $state, 1 );
	}

	return $state_enum[$id];
}

sub dfa {
	my ($self) = @_;

	if ( exists $self->{dfa} ) {
		return $self->{dfa};
	}

	my $dfa        = FLAT::DFA->new();
	my @state_enum = $self->model->get_state_enum;

	$dfa->add_states( scalar @state_enum );
	$dfa->set_starting(0);
	$dfa->set_accepting( $dfa->get_states );

	for my $transition ( $self->model->transitions ) {
		print Dumper( $transition->{parameters} );

		for my $param ( @{ $transition->{parameters} } ) {
			if ( not defined $param->{values} ) {
				die(
"argument values for transition $transition->{name} are undefined\n"
				);
			}
			if ( @{ $param->{values} } == 0 ) {
				die(
"argument-value list for transition $transition->{name} is empty \n"
				);
			}
		}

		my @argtuples
		  = cartesian { 1 } map { $_->{values} } @{ $transition->{parameters} };

		# cartesian will return a one-element list containing a reference to
		# an empty array if @{$transition->{parameters}} is empty

		for my $argtuple (@argtuples) {
			for my $transition_pair ( @{ $transition->{transitions} } ) {
				my ( $origin, $destination ) = @{$transition_pair};
				$dfa->add_transition(
					$self->model->get_state_id($origin),
					$self->model->get_state_id($destination),
					':' . $transition->{name} . '!' . join( '!', @{$argtuple} )
				);
			}
		}
	}

	if ( $self->{excluded_states} and @{ $self->{excluded_states} } ) {
		my @to_delete = map { $self->model->get_state_id($_) }
		  @{ $self->{excluded_states} };
		$dfa->delete_states(@to_delete);
	}

	$self->{dfa} = $dfa;

	say $dfa->as_summary;

	return $dfa;
}

sub run_str_to_trace {
	my ( $self, $run_str ) = @_;
	my @trace;
	my $dfa             = $self->dfa;
	my %param           = $self->model->parameter_hash;
	my $state           = 0;
	my $state_duration  = $self->{state_duration} // 1000;
	my @state_enum      = $self->model->get_state_enum;
	my $prev_transition = {};
	for my $transition_str ( split( qr{ : }x, $run_str ) ) {
		my ( $cmd, @args ) = split( qr{ ! }x, $transition_str );
		my $state_name = $self->reduced_id_to_state($state);
		my $transition = $self->model->get_transition_by_name($cmd);

		push(
			@trace,
			{
				isa  => 'state',
				name => $state_name,
				plan => {
					time => $prev_transition->{timeout}{static}
					  // $state_duration,
					power  => $self->model->get_state_power($state_name),
					energy => $self->model->get_state_power($state_name)
					  * $state_duration,
				},
				parameter => { map { $_ => $param{$_}{value} } keys %param, },
			},
			{
				isa  => 'transition',
				name => $cmd,
				args => [@args],
				code => [ $cmd, @args ],
				plan => {
					level   => $transition->{level},
					energy  => $transition->{energy}{static},
					timeout => $transition->{timeout}{static},
				},
				parameter => { map { $_ => $param{$_}{value} } keys %param, },
			},
		);

		$self->model->update_parameter_hash( \%param, $cmd, @args );

		($state) = $dfa->successors( $state, ":${transition_str}" );

		if ( not defined $state ) {
			die("Transition $transition_str is invalid or has no successors\n");
		}

		$prev_transition = $transition;
		for my $extra_cmd (
			$self->model->get_state_extra_transitions( $state_enum[$state] ) )
		{
			$state_name = $self->reduced_id_to_state($state);
			$transition = $self->model->get_transition_by_name($extra_cmd);
			push(
				@trace,
				{
					isa  => 'state',
					name => $state_name,
					plan => {
						time => $prev_transition->{timeout}{static}
						  // $state_duration,
						power  => $self->model->get_state_power($state_name),
						energy => $self->model->get_state_power($state_name)
						  * $state_duration,
					},
					parameter =>
					  { map { $_ => $param{$_}{value} } keys %param, },
				},
				{
					isa  => 'transition',
					name => $extra_cmd,
					args => [],
					code => [$extra_cmd],
					plan => {
						level   => $transition->{level},
						energy  => $transition->{energy}{static},
						timeout => $transition->{timeout}{static},
					},
					parameter =>
					  { map { $_ => $param{$_}{value} } keys %param, },
				}
			);
			$prev_transition = $transition;
		}
	}

	# required for unscheduled extra states and transitions caused by interrupts
	$trace[-1]{final_parameter}
	  = { map { $_ => $param{$_}{value} } keys %param, };
	return @trace;
}

sub traces {
	my ($self) = @_;

	# Warning: This function is not deterministic!
	# Therefore, results are cached. When in doubt, reload traces / execution
	# plan from DriverEval.json

	if ( exists $self->{traces} ) {
		return @{ $self->{traces} };
	}

	my $max_iter = $self->{trace_revisit} // 2;
	my $next     = $self->dfa->new_deepdft_string_generator($max_iter);
	my $trace_id = 1;

	my ( @raw_runs, @traces );
	my $filter_re;

	if ( $self->{trace_filter} and @{ $self->{trace_filter} } ) {
		my @res;
		for my $filter ( @{ $self->{trace_filter} } ) {
			my $re = $filter;
			$re =~ s{,}{![^:]*:}g;
			$re =~ s{$}{![^:]*)};
			$re =~ s{^}{(^};
			if ( $re =~ m{ \$ }x ) {
				$re =~ s{\$}{};
				$re =~ s{\)$}{\$)};
			}
			push( @res, $re );
		}
		$filter_re = join( q{|}, @res );
	}

	while ( my $run = $next->() ) {
		$run = substr( $run, 1 );
		if ( $filter_re and not $run =~ m{$filter_re} ) {
			next;
		}
		@raw_runs = grep { $_ ne substr( $run, 0, length($_) ) } @raw_runs;
		push( @raw_runs, $run );
	}

	if ( @raw_runs == 0 ) {
		say STDERR "--trace-filter did not match any run. Aborting.";
		exit 1;
	}

	@raw_runs = sort @raw_runs;

	for my $run_str (@raw_runs) {
		my @trace = $self->run_str_to_trace($run_str);
		push(
			@traces,
			{
				id    => $trace_id,
				trace => [@trace],
			}
		);
		$trace_id++;
	}

	$self->{traces} = [@traces];

	return @traces;
}

1;
