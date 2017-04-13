package Kratos::DFADriver;

use strict;
use warnings;
use 5.020;

use parent 'Class::Accessor';

use Archive::Tar;
use AspectC::Repo;
use Carp;
use Carp::Assert::More;
use Cwd;
use DateTime;
use Device::SerialPort;
use File::Slurp qw(read_dir read_file write_file);
use IPC::Run qw(harness);
use JSON;
use Kratos::DFADriver::DFA;
use Kratos::DFADriver::Model;
use List::Util qw(first);
use List::MoreUtils qw(pairwise);
use MIMOSA;
use MIMOSA::Log;

Kratos::DFADriver->mk_ro_accessors(qw(class_name dfa mimosa model repo));

our $VERSION = '0.00';

sub new {
	my ( $class, %opt ) = @_;

	my $self = \%opt;

	$self->{dfa}           = Kratos::DFADriver::DFA->new(%opt);
	$self->{mimosa}        = MIMOSA->new(%opt);
	$self->{model}         = Kratos::DFADriver::Model->new(%opt);
	$self->{repo}          = AspectC::Repo->new;
	$self->{class_name}    = $self->{model}->class_name;
	$self->{lp}{iteration} = 1;

	bless( $self, $class );

	$self->set_paths;
	$self->dfa->set_model( $self->model );

	return $self;
}

sub set_paths {
	my ($self) = @_;

	my $xml_path = $self->{xml_file};
	$xml_path =~ s{ /?+dfa-driver/[^/]+[.]xml $ }{}x;

	my $prefix = $self->{prefix} = cwd() . "/${xml_path}/src";
	my $class_prefix
	  = $self->repo->get_class_path_prefix( $self->{class_name} );
	$self->{ah_file} = "${prefix}/${class_prefix}_dfa.ah";
	$self->{cc_file} = "${prefix}/${class_prefix}_dfa.cc.inc";
	$self->{h_file}  = "${prefix}/${class_prefix}_dfa.h.inc";
}

sub set_output {
	my ($self, $mode) = @_;

	if ($mode eq 'tex') {
		$self->{tex} = 1;
	}

	return $self;
}

sub preprocess {
	my ( $self, @files ) = @_;
	my @logs;
	my @json_files;

	for my $i ( 0 .. $#files ) {
		push(
			@logs,
			MIMOSA::Log->new(
				data_file     => $files[$i],
				fast_analysis => $self->{fast_analysis},
				model         => $self->model,
				merge_args    => $self->{merge_args},
				tmpsuffix     => $i,
			)
		);
	}

	for my $log (@logs) {
		if ( not $self->{cache} or not $log->load_cache ) {
			$log->load_archive;
			$log->preprocess;
			$log->save_cache;
		}
		push( @json_files, $log->json_name );
	}

	$self->{log} = $logs[0];
	return ( \@logs, \@json_files );
}

sub analyze {
	my ( $self, @files )      = @_;
	my ( $logs, $json_files ) = $self->preprocess(@files);
	$self->log->analyze( @{$json_files} );
}

sub validate_model {
	my ( $self, @files )      = @_;
	my ( $logs, $json_files ) = $self->preprocess(@files);
	$self->log->validate( @{$json_files} );
	$self->assess_validation;
}

sub crossvalidate_model {
	my ( $self, @files )      = @_;
	my ( $logs, $json_files ) = $self->preprocess(@files);
	$self->log->crossvalidate( @{$json_files} );
}

sub log {
	my ( $self, $file ) = @_;

	if ($file) {
		$self->{log} = undef;
	}

	$self->{log} //= MIMOSA::Log->new(
		data_file => $file // $self->{data_file},
		fast_analysis => $self->{fast_analysis},
		model         => $self->model,
		merge_args    => $self->{merge_args}
	);

	return $self->{log};
}

sub assess_fits {
	my ( $self, $hash, $param, $funtype ) = @_;

	$funtype //= 'fit_guess';

	my $errmap = $hash->{$funtype}{$param};
	my @errors = map { [ $_, $errmap->{$_} ] } keys %{$errmap};
	@errors = sort { $a->[1]{rmsd} <=> $b->[1]{rmsd} } @errors;

	my $min_err = $errors[0][1]{rmsd};
	@errors = grep { $_->[1]{rmsd} <= 2 * $min_err } @errors;
	my @function_types = map {
		sprintf( '%s (%.f / %.2f%%)', $_->[0], $_->[1]{rmsd}, $_->[1]{smape} )
	} @errors;

	return @function_types;
}

sub printf_aggr {
	my ( $self, $hash, $key, $unit ) = @_;

	$hash = $hash->{$key};

	if ( exists $hash->{median_goodness}{smape} ) {
		printf(
			"  %s: static error: %.2f%% / %.f %s  (σ = %.f)\n",
			$key,
			$hash->{median_goodness}{smape},
			$hash->{median_goodness}{mae},
			$unit, $hash->{std_inner}
		);

#printf("  %s: median %.f (%.2f / %.2f%%), mean %.f (%.2f / %.2f%%), σ %.f %s\n",
#	$key,
#	$hash->{median},
#	$hash->{median_goodness}{mae}   // -1,
#	$hash->{median_goodness}{smape} // -1,
#	$hash->{mean},
#	$hash->{mean_goodness}{mae}   // -1,
#	$hash->{mean_goodness}{smape} // -1,
#	$hash->{std_inner},
#	$unit
#);
	}
	else {
		printf(
			"  %s: static error: %.f %s  (σ = %.f)\n",
			$key, $hash->{median_goodness}{mae},
			$unit, $hash->{std_inner}
		);

		#printf(
		#	"  %s: median %.f (%.2f), mean %.f (%.2f), σ %.f %s\n",
		#	$key, $hash->{median}, $hash->{median_goodness}{mae},
		#	$hash->{mean}, $hash->{mean_goodness}{mae},
		#	$hash->{std_inner}, $unit
		#);
	}
}

sub printf_aggr_tex {
	my ( $self, $hash, $key, $unit, $divisor) = @_;

	$hash = $hash->{$key};

	if ($unit eq 'ms' and $hash->{median} < 1e3) {
		$unit = '\us';
		$divisor = 1;
	}
	elsif ($unit eq '\uJ' and $hash->{median} < 1e6) {
		$unit = 'nJ';
		$divisor = 1e3;
	}
	elsif ($unit eq '\uW' and $hash->{median} >= 1e3) {
		$unit = 'mW';
		$divisor = 1e3;
	}

	use locale;

	printf(' & & \unit[%.3g]{%s}', $hash->{median} / $divisor, $unit);
}

sub printf_count_tex {
	my ( $self, $hash, $key ) = @_;

	if ($hash) {
		$hash = $hash->{$key};

		printf(' & %d', $hash->{count});
	}
	else {
		printf(' & ');
	}
}

sub printf_eval_tex {
	my ( $self, $hash, $key, $unit, $divisor) = @_;

	$hash = $hash->{$key};

	if ($unit eq 'ms' and $hash->{median_goodness}{mae} < 1e3) {
		$unit = '\us';
		$divisor = 1;
	}
	if ($unit eq '\uJ' and $hash->{median_goodness}{mae} < 1e6) {
		$unit = 'nJ';
		$divisor = 1e3;
	}

	use locale;

	printf("\n%20s & \\unit[%.3g]{%s} & \\unit[%.2g]{\\%%}",
		q{},
		$hash->{median_goodness}{mae} / $divisor, $unit,
		$hash->{median_goodness}{smape} // -1
	);
}

sub printf_goodness {
	my ( $self, $modval, $hash, $key, $unit ) = @_;

	$hash = $hash->{$key};

	if ( exists $hash->{goodness}->{smape} ) {
		printf(
"  %s: model %.f %s, log ~=%.f / µ=%.f %s, mean absolute error %.2f %s (%.2f%%)\n",
			$key,                     $modval,
			$unit,                    $hash->{median},
			$hash->{mean},            $unit,
			$hash->{goodness}->{mae}, $unit,
			$hash->{goodness}{smape}
		);
	}
	else {
		printf(
"  %s: model %.f %s, log ~=%.f / µ=%.f %s, mean absolute error %.2f %s\n",
			$key, $modval, $unit, $hash->{median}, $hash->{mean}, $unit,
			$hash->{goodness}->{mae}, $unit );
	}
}

sub printf_online_goodness {
	my ( $self, $hash, $key, $unit ) = @_;

	$hash = $hash->{$key};

	if ( exists $hash->{goodness}->{smape} ) {
		printf(
"  %s: ~=%.f / µ=%.f %s, mean absolute error %.2f %s (%.2f%%)\n",
			$key,                     $hash->{median},
			$hash->{mean},            $unit,
			$hash->{goodness}->{mae}, $unit,
			$hash->{goodness}{smape}
		);
	}
	else {
		printf(
"  %s: ~=%.f / µ=%.f %s, mean absolute error %.2f %s\n",
			$key, $hash->{median}, $hash->{mean}, $unit,
			$hash->{goodness}->{mae}, $unit );
	}
}

sub printf_clip {
	my ( $self, $hash ) = @_;

	if ( $hash->{clip}{max} > 0.01 ) {
		printf(
			"  WARNING: Up to %.f%% clipping in power measurements (avg %.f%%)"
			  . ", results are unreliable\n",
			$hash->{clip}{max} * 100,
			$hash->{clip}{mean} * 100
		);
	}
}

sub printf_parameterized {
	my ( $self, $hash, $key ) = @_;
	$hash = $hash->{$key};

	my $std_global    = $hash->{std_inner};
	my $std_ind_arg   = $hash->{std_arg};
	my $std_ind_param = $hash->{std_param};
	my $std_ind_trace = $hash->{std_trace};
	my $std_by_arg    = $hash->{std_by_arg} // {};
	my $std_by_param  = $hash->{std_by_param};
	my $std_by_trace  = $hash->{std_by_trace} // {};
	my $arg_ratio;
	my $param_ratio;
	my $trace_ratio;

	if ( $std_global > 0 ) {
		$param_ratio = $std_ind_param / $std_global;
		if (defined $std_ind_arg) {
			$arg_ratio = $std_ind_arg / $std_global;
		}
	}
	if ( $std_ind_param > 0) {
		$trace_ratio = $std_ind_trace / $std_ind_param;
	}

	if (    $std_global > 10
		and $param_ratio < 0.5
		and not exists $hash->{function}{user} )
	{
		printf( "  %s: should be parameterized (%.2f / %.2f = %.3f)\n",
			$key, $std_ind_param, $std_global, $param_ratio );
	}
	if (
		(
			   $std_global < 10
			or $param_ratio > 0.5
		)
		and exists $hash->{function}{user}
	  )
	{
		printf( "  %s: should not be parameterized (%.2f / %.2f = %.3f)\n",
			$key, $std_ind_param, $std_global,
			$param_ratio ? $param_ratio : 0 );
	}

	if ( defined $std_ind_arg and   $std_global > 10
		and $arg_ratio < 0.5
		and not exists $hash->{function}{user_arg} )
	{
		printf( "  %s: depends on arguments (%.2f / %.2f = %.3f)\n",
			$key, $std_ind_arg, $std_global, $arg_ratio );
	}
	if ( defined $std_ind_arg and
		(
			   $std_global < 10
			or $arg_ratio > 0.5
		)
		and exists $hash->{function}{user_arg}
	  )
	{
		printf( "  %s: should not depend on arguments (%.2f / %.2f = %.3f)\n",
			$key, $std_ind_arg, $std_global,
			$arg_ratio ? $arg_ratio : 0 );
	}

	if ( $std_global > 10 and $trace_ratio < 0.5 ) {
		printf(
			"  %s: model insufficient, depends on trace (%.2f / %.2f = %.3f)\n",
			$key, $std_ind_trace, $std_ind_param, $trace_ratio );
	}

	if ( $std_global < 10 ) {
		return;
	}

	for my $param ( sort keys %{$std_by_param} ) {
		my $std_this = $std_by_param->{$param};
		my $ratio    = $std_ind_param / $std_this;
		my $status   = 'does not depend';
		my $fline    = q{};
		if ( $ratio < 0.6 ) {
			$status = 'might depend';
			$fline  = q{, probably };
			$fline .= join( ' or ', $self->assess_fits( $hash, $param ) );
		}
		if ( $ratio < 0.3 ) {
			$status = 'depends';
		}
		if ($fline) {
			printf( "  %s: %s on global %s (%.2f / %.2f = %.3f%s)\n",
				$key, $status, $param, $std_ind_param, $std_this, $ratio,
				$fline );
		}
	}

	for my $arg ( sort keys %{$std_by_arg} ) {
		my $std_this = $std_by_arg->{$arg};
		my $ratio    = $std_ind_arg / $std_this;
		my $status   = 'does not depend';
		my $fline    = q{};
		if ( $ratio < 0.6 ) {
			$status = 'might depend';
			$fline  = q{, probably };
			$fline .= join( ' or ', $self->assess_fits( $hash, $arg, 'arg_fit_guess' ) );
		}
		if ( $ratio < 0.3 ) {
			$status = 'depends';
		}
		if ($fline) {
			printf( "  %s: %s on local %s (%.2f / %.2f = %.3f%s)\n",
				$key, $status, $arg, $std_ind_arg, $std_this, $ratio,
				$fline );
		}
	}

	for my $transition ( sort keys %{$std_by_trace} ) {
		my $std_this = $std_by_trace->{$transition};
		my $ratio = $std_ind_trace / $std_this;
		if ($ratio < 0.4) {
			printf(
"  %s: depends on presence of %s in trace (%.2f / %.2f = %.3f)\n",
				$key, $transition, $std_ind_trace, $std_this, $ratio );
		}
	}
}

sub printf_fit {
	my ( $self, $hash, $key, $unit ) = @_;
	$hash = $hash->{$key};

	for my $funtype (sort keys %{$hash->{function}}) {
		if ( exists $hash->{function}{$funtype}{error} ) {
			printf( "  %s: %s function could not be fitted: %s\n",
				$key, $funtype, $hash->{function}{$funtype}{error} );
		}
		else {
			printf(
				"  %s: %s function fit error: %.2f%% / %.f %s\n",
				$key, $funtype,
				$hash->{function}{$funtype}{fit}{smape} // -1,
				$hash->{function}{$funtype}{fit}{mae}, $unit
			);
		}
	}

	for my $pair (['param_mean_goodness', 'param mean/ssr-fit'],
			['param_median_goodness', 'param median/static'],
			['arg_mean_goodness', 'arg mean/ssr-fit'],
			['arg_median_goodness', 'arg median/static']) {
		my ($goodness, $desc) = @{$pair};
		if ( exists $hash->{$goodness} ) {
			printf(
				"  %s: %s LUT error: %.2f%% / %.f %s / %.f\n",
				$key, $desc,
				$hash->{$goodness}{smape} // -1,
				$hash->{$goodness}{mae}, $unit,
				$hash->{$goodness}{rmsd}
			);
		}
	}
}

sub assess_model {
	my ($self) = @_;

	for my $name ( sort keys %{ $self->{log}{aggregate}{state} } ) {
		my $state = $self->{log}{aggregate}{state}{$name};

		printf( "Assessing %s:\n", $name );

		$self->printf_clip($state);
		$self->printf_aggr( $state, 'power', 'µW' );
		$self->printf_parameterized( $state, 'power' );
		$self->printf_fit( $state, 'power', 'µW' );
	}
	for my $name ( sort keys %{ $self->{log}{aggregate}{transition} } ) {
		my $transition = $self->{log}{aggregate}{transition}{$name};

		printf( "Assessing %s:\n", $name );

		$self->printf_clip($transition);
		$self->printf_aggr( $transition, 'duration',        'µs' );
		$self->printf_parameterized( $transition, 'duration' );
		$self->printf_fit( $transition, 'duration', 'µs' );
		$self->printf_aggr( $transition, 'energy', 'pJ' );
		$self->printf_parameterized( $transition, 'energy' );
		$self->printf_fit( $transition, 'energy', 'pJ' );
		$self->printf_aggr( $transition, 'rel_energy_prev', 'pJ' );
		$self->printf_parameterized( $transition, 'rel_energy_prev' );
		$self->printf_fit( $transition, 'rel_energy_prev', 'pJ' );

		if ( exists $transition->{rel_energy_next}{median} ) {
			$self->printf_aggr( $transition, 'rel_energy_next', 'pJ' );
			$self->printf_parameterized( $transition, 'rel_energy_next' );
			$self->printf_fit( $transition, 'rel_energy_next', 'pJ' );
		}

		if ( exists $transition->{timeout}{median} ) {
			$self->printf_aggr( $transition, 'timeout', 'µs' );
			$self->printf_parameterized( $transition, 'timeout' );
			$self->printf_fit( $transition, 'timeout', 'µs' );
		}
	}

}

sub assess_model_tex {
	my ($self) = @_;
	say '\begin{tabular}{|c|rrr|r|}\\hline';
	say 'Zustand & $\MmedP$ & & & $n$ \\\\\\hline';
	for my $name ( sort keys %{ $self->{log}{aggregate}{state} } ) {
		my $state = $self->{log}{aggregate}{state}{$name};

		printf("\n%20s", $name);

		$self->printf_aggr_tex( $state, 'power', '\uW', 1 );
		$self->printf_eval_tex( $state, 'power', '\uW', 1 );
		$self->printf_count_tex( $state, 'power' );
		print " \\\\";
	}
	say '\end{tabular}\\\\';
	say '\vspace{0.5cm}';
	say '\begin{tabular}{|c|rr|rr|rr|r|}\\hline';
	say 'Transition & & $\MmedE$ & & $\MmedF$ & & $\Mmeddur$ & $n$ \\\\\\hline';
	for my $name ( sort keys %{ $self->{log}{aggregate}{transition} } ) {
		my $transition = $self->{log}{aggregate}{transition}{$name};

		printf("\n%20s", $name);

		$self->printf_aggr_tex( $transition, 'energy', '\uJ', 1e6 );
		$self->printf_aggr_tex( $transition, 'rel_energy_prev', '\uJ', 1e6 );
		$self->printf_aggr_tex( $transition, 'rel_energy_next', '\uJ', 1e6 );
		$self->printf_aggr_tex( $transition, 'duration', 'ms', 1e3 );
		$self->printf_count_tex( $transition, 'energy' );
		print " \\\\";
		$self->printf_eval_tex( $transition, 'energy', '\uJ', 1e6 );
		$self->printf_eval_tex( $transition, 'rel_energy_prev', '\uJ', 1e6 );
		$self->printf_eval_tex( $transition, 'rel_energy_next', '\uJ', 1e6 );
		$self->printf_eval_tex( $transition, 'duration', 'ms', 1e3 );
		$self->printf_count_tex;
		print " \\\\";
	}
	print "\\hline\n";
	say '\end{tabular}';
}

sub assess_validation {
	my ($self) = @_;

	for my $name ( sort keys %{ $self->{log}{aggregate}{state} } ) {
		my $state = $self->{log}{aggregate}{state}{$name};

		printf( "Validating %s:\n", $name );
		$self->printf_clip($state);
		$self->printf_goodness( $self->model->get_state_power($name),
			$state, 'power', 'µW' );
		$self->printf_fit( $state, 'power', 'µW' );
		$self->printf_online_goodness(
			$state, 'online_power', 'µW' );
		$self->printf_online_goodness(
			$state, 'online_duration', 'µs' );
	}
	for my $name ( sort keys %{ $self->{log}{aggregate}{transition} } ) {
		my $transition = $self->{log}{aggregate}{transition}{$name};

		printf( "Validating %s:\n", $name );
		$self->printf_clip($transition);
		$self->printf_goodness(
			$self->model->get_transition_by_name($name)->{duration}{static},
			$transition, 'duration', 'µs' );
		$self->printf_goodness(
			$self->model->get_transition_by_name($name)->{energy}{static},
			$transition, 'energy', 'pJ' );
		$self->printf_goodness(
			$self->model->get_transition_by_name($name)->{rel_energy_prev}{static},
			$transition, 'rel_energy_prev', 'pJ' );
		if ( exists $transition->{rel_energy_next}{median} ) {
			$self->printf_goodness(
				$self->model->get_transition_by_name($name)->{rel_energy_next}{static},
				$transition, 'rel_energy_next', 'pJ' );
		}
		if ( exists $transition->{timeout}{median} ) {
			$self->printf_fit( $transition, 'timeout', 'µs' );
		}
	}
}

sub update_model {
	my ($self) = @_;

	for my $name (sort keys %{ $self->{log}{aggregate}{state} }) {
		my $state = $self->{log}{aggregate}{state}{$name};
		$self->model->set_state_power( $name, $state->{power}{median} );
		for my $fname ( keys %{ $state->{power}{function} } ) {
			$self->model->set_state_params(
				$name, $fname,
				$state->{power}{function}{$fname}{raw},
				@{ $state->{power}{function}{$fname}{params} }
			);
		}
		if ($self->{with_lut}) {
			$self->model->set_state_lut( $name, 'power', $state->{power}{median_by_param} );
		}
	}
	for my $name (sort keys %{ $self->{log}{aggregate}{transition} }) {
		my $transition = $self->{log}{aggregate}{transition}{$name};
		my @keys = (qw(duration energy rel_energy_prev rel_energy_next));

		if ($self->model->get_transition_by_name($name)->{level} eq 'epilogue') {
			push(@keys, 'timeout');
		}

		for my $key (@keys) {
			$self->model->set_transition_property(
				$name, $key, $transition->{$key}{median}
			);
			for my $fname ( keys %{ $transition->{$key}{function} } ) {
				$self->model->set_transition_params(
					$name, $key, $fname,
					$transition->{$key}{function}{$fname}{raw},
					@{ $transition->{$key}{function}{$fname}{params} }
				);
			}
			if ($self->{with_lut}) {
				$self->model->set_transition_lut( $name, $key, $transition->{$key}{median_by_param} );
			}
		}
	}

	$self->model->save;
}

sub reset_model {
	my ($self) = @_;

	$self->model->reset;
	$self->model->save;
}

sub to_ah {
	my ($self)       = @_;
	my $class_name   = $self->{class_name};
	my $repo         = $self->repo;
	my $class_header = $repo->{class}{$class_name}{sources}[0]{file};

	my @transition_names
	  = grep { $_ ne q{?} } map { $_->{name} } $self->model->transitions;

	my $trigger_port = $self->{trigger_port};
	my $trigger_pin  = $self->{trigger_pin};

	my $ignore_nested = q{};
	my $adv_type      = 'execution';

	if ( $self->{ignore_nested} ) {
		$adv_type      = 'call';
		$ignore_nested = "&& !within(\"${class_name}\")";
	}

	my $ah_buf = <<"EOF";

#ifndef ${class_name}_DFA_AH
#define ${class_name}_DFA_AH

#include "drivers/dfa_driver.h"
#include "drivers/gpio.h"
#include "drivers/eUSCI_A/uart/prototype_uart.h"
#include "${class_header}"

pointcut InnerTransition() = execution("% ${class_name}::%(...)");

EOF

	if ( defined $trigger_port and defined $trigger_pin ) {

		$ah_buf .= "aspect ${class_name}_Trigger {\n\n";

		$ah_buf .= 'pointcut Transition() = "'
		  . join( q{" || "},
			map { "% ${class_name}::$_(...)" } @transition_names )
		  . "\";\n\n";
		$ah_buf .= <<"EOF";

		advice execution("void initialize_devices()") : after() {
			setOutput(${trigger_port}, ${trigger_pin});
		}

		advice ${adv_type}(Transition()) ${ignore_nested} : before() {
			pinHigh(${trigger_port}, ${trigger_pin});
		}
		advice ${adv_type}(Transition()) ${ignore_nested} : after() {
			/* 22 = 10.2us delay @ 16MHz */
			/* 32 = 14.6us delay @ 16MHz */
			/* 64 = 28.6us delay @ 16MHz */
			/* 160 = 50.6us delay @ 16MHz */
			for (unsigned int i = 0; i < 64; i++)
				asm volatile("nop");
			pinLow(${trigger_port}, ${trigger_pin});
		}

		advice execution(Transition()) : order("${class_name}_DFA", "${class_name}_Trigger");

EOF

		if ( $self->{ignore_nested} ) {
			for my $transition ( $self->model->transitions ) {
				if ( $transition->{level} eq 'epilogue' ) {
					$ah_buf .= <<"EOF";

		advice execution("% ${class_name}::$transition->{name}(...)") : before() {
			pinHigh(${trigger_port}, ${trigger_pin});
		}
		advice execution("% ${class_name}::$transition->{name}(...)") : after() {
			for (unsigned int i = 0; i < 64; i++)
				asm volatile("nop");
			pinLow(${trigger_port}, ${trigger_pin});
		}

EOF
				}
			}
		}
		$ah_buf .= "};\n\n";
	}

	$ah_buf .= "aspect ${class_name}_DFA {\n\n";

	for my $transition ( $self->model->transitions ) {
		if ( $transition->{name} ne q{?} ) {
			my $dest_state_id
			  = $self->model->get_state_id( $transition->{destination} );
			if ( $transition->{level} eq 'user' ) {
				$ah_buf .= <<"EOF";

		advice ${adv_type}("% ${class_name}::$transition->{name}(...)") ${ignore_nested} : after() {
			tjp->target()->passTransition(${class_name}::statepower[tjp->target()->state],
			$transition->{rel_energy_prev}{static}, $transition->{id},
			${dest_state_id});
		};

EOF
			}
			else {
				$ah_buf .= <<"EOF";

		advice execution("% ${class_name}::$transition->{name}(...)") : after() {
			tjp->target()->passTransition(${class_name}::statepower[tjp->target()->state],
			$transition->{rel_energy_prev}{static}, $transition->{id},
			${dest_state_id});
		};

EOF
			}
		}
	}

	$ah_buf .= <<"EOF";

};
#endif

EOF

	return $ah_buf;
}

sub to_cc {
	my ($self) = @_;
	my $class_name = $self->{class_name};

	my @state_enum = $self->model->get_state_enum;

	my $buf
	  = "DFA_Driver::power_uW_t ${class_name}::statepower[] = {"
	  . join( ', ', map { $self->model->get_state_power($_) } @state_enum )
	  . "};\n";

	return $buf;
}

sub to_h {
	my ($self) = @_;

	my @state_enum = $self->model->get_state_enum;

	my $buf
	  = "public:\n"
	  . "static power_uW_t statepower[];\n"
	  . "enum State : uint8_t {"
	  . join( ', ', @state_enum ) . "};\n";

	return $buf;
}

sub to_tikz {
	my ($self) = @_;

	my $buf = <<'EOF';

	\begin{tikzpicture}[node distance=3cm,>=stealth',bend angle=45,auto,->]
		\tikzstyle{state}=[ellipse,thick,draw=black!75,minimum size=1cm,inner sep=2pt]

EOF

	my @state_enum = $self->model->get_state_enum;
	my $initial    = shift(@state_enum);
	my $prev       = $initial;
	my $ini_name   = $initial;

	if ( $ini_name eq 'UNINITIALIZED' ) {
		$ini_name = '?';
	}

	$buf
	  .= "\t\t\\node [state,initial,initial text={},initial where=left] ($initial) {\\small $ini_name};\n";
	for my $state (@state_enum) {
		$buf
		  .= "\t\t\\node [state,right of=${prev}] ($state) {\\small $state};\n";
		$prev = $state;
	}

	$buf .= "\n\t\t\\path\n";

	for my $transition ( $self->model->transitions ) {
		for my $origin ( @{ $transition->{origins} } ) {
			my @edgestyles;
			if ( $transition->{level} eq 'epilogue' ) {
				push( @edgestyles, 'dashed' );
			}
			if ( $origin eq $transition->{destination} ) {
				push( @edgestyles, 'loop above' );
			}
			my $edgestyle
			  = @edgestyles ? '[' . join( q{,}, @edgestyles ) . ']' : q{};
			$buf
			  .= "\t\t  ($origin) edge ${edgestyle} node {$transition->{name}} ($transition->{destination})\n";
		}
	}
	$buf .= "\t\t;\n";
	$buf .= "\t\\end{tikzpicture}\n";

	return $buf;
}

sub to_test_ah {
	my ($self) = @_;

	my $buf = <<'EOF';

/*
 * Autogenerated code -- Manual changes are not preserved
 * vim:readonly
 */

#ifndef DRIVEREVAL_AH
#define DRIVEREVAL_AH

#include "DriverEval.h"
#include "syscall/guarded_scheduler.h"

aspect StartDFADriverEvalThread {
	advice execution("void ready_threads()") : after() {
		organizer.Scheduler::ready(driverEvalThread);
	}
};

#endif

EOF

	return $buf;
}

sub to_test_cc {
	my ($self) = @_;

	my @runs       = $self->dfa->traces;
	my @state_enum = $self->model->get_state_enum;
	my $dfa        = $self->dfa->dfa;
	my $num_runs   = @runs;
	my $instance   = $self->repo->get_class_instance( $self->{class_name} );

	my $state_duration = $self->{state_duration} // 1000;

	my $buf = <<"EOF";

/*
 * Autogenerated code - Manual changes are not preserved.
 * vim:readonly
 */

#include "DriverEval.h"
#include "syscall/guarded_buzzer.h"

DeclareThread(DriverEvalThread, driverEvalThread, 256);

EOF

	$buf .= $self->model->heap_code;

	$buf .= <<"EOF";
void DriverEvalThread::action()
{
	Guarded_Buzzer buzzer;

	while (1) {

		/* wait for MIMOSA calibration */
		buzzer.sleep(12000);
		buzzer.set(${state_duration});


EOF

	$buf .= $self->model->startup_code;
	$buf .= "${instance}.startIteration(${num_runs});\n";

	for my $run (@runs) {
		$buf .= "\t\t/* test run $run->{id} start */\n";
		$buf .= "\t\t${instance}.resetLogging();\n";
		# $buf .= "\t\t${instance}.resetAccounting();\n"; # TODO sinnvoll?
		my $state = 0;
		for my $transition ( grep { $_->{isa} eq 'transition' }
			@{ $run->{trace} } )
		{
			my ( $cmd, @args ) = @{ $transition->{code} };
			my ($new_state)
			  = $dfa->successors( $state, ":${cmd}!" . join( '!', @args ) );
			my $state_name     = $self->dfa->reduced_id_to_state($state);
			my $new_state_name = $self->dfa->reduced_id_to_state($new_state);
			$buf .= "\t\t/* Transition $state_name -> $new_state_name */\n";

			if ( $self->model->get_transition_by_name($cmd)->{level} eq
				'epilogue' )
			{
				$buf .= "\t\t/* wait for $cmd interrupt */\n";
				$buf .= "\t\tbuzzer.sleep();\n";
			}
			else {
				$buf .= sprintf( "\t\t%s.%s(%s);\n",
					$instance, $cmd, join( ', ', @args ) );
				$buf .= "\t\tbuzzer.sleep();\n";
			}
			$buf .= $self->model->after_transition_code;
			$state = $new_state;
		}
		$buf .= "\t\t${instance}.dumpLog();\n\n";
	}

	$buf .= $self->model->shutdown_code;
	$buf .= "${instance}.stopIteration(); }}\n";

	return $buf;
}

sub to_test_h {
	my ($self) = @_;

	my $class_prefix
	  = $self->repo->get_class_path_prefix( $self->{class_name} );

	my $buf = <<"EOF";

/*
 * Autogenerated code -- Manual changes are not preserved
 * vim:readonly
 */

#ifndef DRIVEREVAL_H
#define DRIVEREVAL_H

#include "${class_prefix}.h"
#include "syscall/thread.h"

class DriverEvalThread : public Thread {
	public:
		DriverEvalThread(void* tos) : Thread(tos) { }
		void action();
};

extern DriverEvalThread driverEvalThread;

#endif

EOF

	return $buf;
}

sub to_test_json {
	my ($self) = @_;

	return JSON->new->encode( [ $self->dfa->traces ] );
}

sub rm_acc_files {
	my ($self) = @_;

	for my $file ( $self->{ah_file}, $self->{cc_file}, $self->{h_file} ) {
		if ( -e $file ) {
			unlink($file);
		}
	}

	return $self;
}

sub write_test_files {
	my ($self) = @_;

	my $prefix = $self->{prefix} . '/apps/DriverEval';

	if ( not -d $prefix ) {
		mkdir($prefix);
	}

	write_file( "${prefix}/DriverEval.ah",   $self->to_test_ah );
	write_file( "${prefix}/DriverEval.cc",   $self->to_test_cc );
	write_file( "${prefix}/DriverEval.h",    $self->to_test_h );
	write_file( "${prefix}/DriverEval.json", $self->to_test_json );

	# Old log may no longer apply to new test files
	unlink("${prefix}/DriverLog.txt");

	return $self;
}

sub rm_test_files {
	my ($self) = @_;

	my $prefix = $self->{prefix} . '/apps/DriverEval/DriverEval';

	for my $file ( "${prefix}.ah", "${prefix}.cc", "${prefix}.h" ) {
		if ( -e $file ) {
			unlink($file);
		}
	}

	return $self;
}

sub archive_files {
	my ($self) = @_;

	$self->{lp}{timestamp} //= DateTime->now( time_zone => 'Europe/Berlin' )
	  ->strftime('%Y%m%d_%H%M%S');

	my $tar = Archive::Tar->new;

	my @eval_files = (
		( map { "src/apps/DriverEval/DriverEval.$_" } (qw(ah cc h json)) ),
		( map { "src/apps/DriverEval/DriverLog.$_" }  (qw(json txt)) ),
	);

	my @mim_files = grep { m{ \. mim }x } read_dir('.');

	$tar->add_files( $self->{xml_file}, @eval_files, @mim_files );

	$tar->add_data(
		'setup.json',
		JSON->new->encode(
			{
				excluded_states => $self->{excluded_states},
				ignore_nested   => $self->{ignore_nested},
				mimosa_offset   => $self->{mimosa_offset},
				mimosa_shunt    => $self->{mimosa_shunt},
				mimosa_voltage  => $self->{mimosa_voltage},
				state_duration  => $self->{state_duration},
				trace_filter    => $self->{trace_filter},
				trace_revisit   => $self->{trace_revisit},
				trigger_pin     => $self->{trigger_pin},
				trigger_port    => $self->{trigger_port},
			}
		)
	);

	$tar->write("../data/$self->{lp}{timestamp}_$self->{class_name}.tar");

	return $self;
}

sub write_acc_files {
	my ($self) = @_;

	write_file( $self->{ah_file}, $self->to_ah );
	write_file( $self->{cc_file}, $self->to_cc );
	write_file( $self->{h_file},  $self->to_h );

	return $self;
}

sub launchpad_connect {
	my ($self) = @_;

	$self->{port_file} //= '/dev/ttyACM1';
	$self->{port} = Device::SerialPort->new( $self->{port_file} )
	  or croak("Error openig serial port $self->{port_file}");

	$self->{port}->baudrate(115200);
	$self->{port}->databits(8);
	$self->{port}->parity('none');
	$self->{port}->read_const_time(500);

	return $self;
}

sub launchpad_flash {
	my ($self) = @_;

	my ( $make_buf, $prog_buf );

	my $remake = harness(
		[ 'make', '-B' ],
		'<'  => \undef,
		'>&' => \$make_buf,
	);

	my $make_program = harness(
		[ 'make', 'program' ],
		'<'  => \undef,
		'>&' => \$prog_buf,
	);

	$remake->run
	  or croak( 'make -B returned ' . $remake->full_result );
	$make_program->run
	  or croak( 'make program returned ' . $remake->full_result );

	return $self;
}

sub launchpad_reset {
	my ($self) = @_;

	my $output_buffer;
	my $make_reset = harness(
		[ 'make', 'reset' ],
		'<'  => \undef,
		'>&' => \$output_buffer,
	);

	$make_reset->run
	  or croak( 'make reset returned ' . $make_reset->full_result );

	return $self;
}

sub launchpad_log_clean {
	my ($self) = @_;

	for my $file ( read_dir('.') ) {
		if ( $file =~ m{ \. mim $ }x ) {
			unlink($file);
		}
	}
}

sub launchpad_log_init {
	my ($self) = @_;

	$self->{lp}{run_id}      = 0;
	$self->{lp}{sync}        = 0;
	$self->{lp}{calibrating} = 0;
	$self->{lp}{run_done}    = 0;
	$self->{lp}{run}         = [];
	$self->{lp}{log}         = [];
	$self->{lp}{errors}      = [];
	$self->{lp}{log_buf}     = q{};

	$self->{lp}{re}{iter_start} = qr{
		^ \[ EP \] \s iteration \s start, \s (?<runs> \d+ ) \s runs $
	}x;
	$self->{lp}{re}{iter_stop} = qr{
		^ \[ EP \] \s iteration \s stop $
	}x;
	$self->{lp}{re}{run_start} = qr{
		^ \[ EP \] \s run \s start $
	}x;
	$self->{lp}{re}{run_stop} = qr{
		^ \[ EP \] \s run \s stop, \s energyUsed = (?<total_e> \S+) $
	}x;
	$self->{lp}{re}{transition} = qr{
		^ \[ EP \] \s dt = (?<delta_t> \S+) \s de = (?<delta_e> \S+) \s
		oldst = (?<old_state> \S+ ) \s trid = (?<transition_id> \S+ ) $
	}x;

	$self->launchpad_connect;

	return $self;
}

sub launchpad_run_done {
	my ($self) = @_;

	if ( $self->{lp}{run_done} ) {
		$self->{lp}{run_done} = 0;
		return 1;
	}
	return 0;
}

sub launchpad_get_errors {
	my ($self) = @_;

	my @errors = @{ $self->{lp}{errors} };
	$self->{lp}{errors} = [];
	return @errors;
}

sub launchpad_log_is_synced {
	my ($self) = @_;

	return $self->{lp}{sync};
}

sub launchpad_log_status {
	my ($self) = @_;

	return ( $self->{lp}{iteration}, $self->{lp}{run_id},
		$self->{lp}{num_runs} );
}

sub launchpad_log_read {
	my ($self) = @_;

	my $port = $self->{port};

	my ( $count, $chars ) = $port->read(1024);

	$self->{lp}{log_buf} .= $chars;

	if ( not defined $count ) {
		$port->close;
		croak("Serial port was disconnected");
	}
	if ( $count > 0 ) {
		my @lines = split( /\n\r/, $chars );
		for my $line (@lines) {
			$self->launchpad_parse_line($line);
		}
	}
}

sub merged_json {
	my ($self) = @_;

	my @traces = $self->dfa->traces;

	for my $run ( @{ $self->{lp}{log} } ) {
		my $trace_idx = $run->{id} - 1;
		my $idx       = 0;

		assert_is( $traces[$trace_idx]{id}, $run->{id} );
		push(@{$traces[$trace_idx]{total_energy}}, $run->{total_energy});
		for my $online_obj ( @{ $run->{trace} } ) {
			my $plan_obj = $traces[$trace_idx]{trace}[$idx];

			#printf("%-15s %-15s\n", $plan_obj->{name}, $online_obj->{name});

			if ( not defined $plan_obj->{name} ) {

				# The planned test run is done, but the hardware reported an
				# epilogue-level transition before the next run was started.

				$traces[$trace_idx]{trace}[$idx] = {
					isa  => $online_obj->{isa},
					name => $online_obj->{name},
					parameter =>
					  $traces[$trace_idx]{trace}[ $idx - 1 ]{parameter},
				};
				if (
					exists $traces[$trace_idx]{trace}[ $idx - 1 ]
					{final_parameter} )
				{
					$traces[$trace_idx]{trace}[$idx]{parameter}
					  = $traces[$trace_idx]{trace}[ $idx - 1 ]{final_parameter};
				}
			}
			else {
				if ($online_obj->{isa} ne $plan_obj->{isa}) {
					printf("Log merge: ISA mismatch (should be %s, is %s) at index %d#%d\n",
						$plan_obj->{isa}, $online_obj->{isa}, $trace_idx, $idx);
					$self->mimosa->kill;
					exit(1);
				}
				if ( $plan_obj->{name} ne 'UNINITIALIZED' ) {
					if ($online_obj->{name} ne $plan_obj->{name}) {
						printf("Log merge: name mismatch (should be %s, is %s) at index %d#%d\n",
						$plan_obj->{name}, $online_obj->{name}, $trace_idx, $idx);
						$self->mimosa->kill;
						exit(1);
					}
				}
			}

			push(
				@{ $traces[$trace_idx]{trace}[$idx]{online} },
				$online_obj->{online}
			);

			$idx++;
		}
	}

	$self->{lp}{log} = [];

	return @traces;
}

sub launchpad_parse_line {
	my ( $self, $line ) = @_;

	if ( $line =~ $self->{lp}{re}{iter_start} ) {
		$self->{lp}{sync}        = 1;
		$self->{lp}{run_id}      = 0;
		$self->{lp}{num_runs}    = $+{runs};
		$self->{lp}{calibrating} = 0;
	}
	elsif ( not $self->{lp}{sync} ) {
		return;
	}
	elsif ( $line =~ $self->{lp}{re}{iter_stop} ) {
		$self->{lp}{iteration}++;
		$self->{lp}{calibrating} = 1;
		write_file( '../kratos/src/apps/DriverEval/DriverLog.txt',
			$self->{lp}{log_buf} );
		write_file(
			'../kratos/src/apps/DriverEval/DriverLog.json',
			JSON->new->encode( [ $self->merged_json ] )
		);
	}
	elsif ( $line =~ $self->{lp}{re}{run_start} ) {
		$self->{lp}{run_id}++;
		$self->{lp}{run} = [];
	}
	elsif ( $line =~ $self->{lp}{re}{run_stop} ) {
		$self->{lp}{run_done} = 1;
		push(
			@{ $self->{lp}{log} },
			{
				id    => $self->{lp}{run_id},
				trace => [ @{ $self->{lp}{run} } ],
				total_energy => 0 + $+{total_e},
			}
		);
	}
	elsif ( $line =~ $self->{lp}{re}{transition} ) {
		push(
			@{ $self->{lp}{run} },
			{
				isa    => 'state',
				name   => ( $self->model->get_state_enum )[ $+{old_state} ],
				online => {
					time   => 0 + $+{delta_t},
					energy => 0 + $+{delta_e},
					power  => 0 + $+{delta_e} / $+{delta_t},
				},
			},
			{
				isa => 'transition',
				name =>
				  $self->model->get_transition_by_id( $+{transition_id} )
				  ->{name},
				online => {
					timeout => 0 + $+{delta_t},
				},
			},
		);
	}
	else {
		$self->{lp}{sync} = 0;
		push( @{ $self->{lp}{errors} }, "Cannot parse $line" );
	}

}

1;
