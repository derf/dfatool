package MIMOSA::Log;

use strict;
use warnings;
use 5.020;

use Archive::Tar;
use Carp;
use File::Slurp qw(read_file read_dir write_file);
use JSON;
use List::Util qw(sum);

#use Statistics::Basic::Mean;
#use Statistics::Basic::StdDev;

our $VERSION = '0.00';
my $CACHE_VERSION = 6;

sub new {
	my ( $class, %opt ) = @_;

	my $self = \%opt;

	$self->{tmpdir} = "/tmp/kratos-dfa-mim-$$";

	if ( $opt{tmpsuffix} ) {
		$self->{tmpdir} .= "-$opt{tmpsuffix}";
	}

	bless( $self, $class );

	return $self;
}

sub tar {
	my ($self) = @_;

	$self->{tar} //= Archive::Tar->new( $self->{data_file} );

	return $self->{tar};
}

sub setup {
	my ($self) = @_;

	return $self->{setup};
}

sub load {
	return new(@_);
}

sub DESTROY {
	my ($self) = @_;

	if ( -d $self->{tmpdir} ) {
		for my $file ( read_dir( $self->{tmpdir} ) ) {
			unlink("$self->{tmpdir}/$file");
		}
		rmdir( $self->{tmpdir} );
	}
}

sub load_archive {
	my ($self) = @_;

	my $tmpdir = $self->{tmpdir};

	my @filelist   = $self->tar->list_files;
	my @mim_files  = sort grep { m{ \. mim $ }x } @filelist;
	my @json_files = map { ( split( qr{[.]}, $_ ) )[0] . '.json' } @mim_files;

	if ( $self->{fast_analysis} ) {
		splice( @mim_files,  4 );
		splice( @json_files, 4 );
	}

	$self->{filelist}    = [@filelist];
	$self->{mim_files}   = [@mim_files];
	$self->{mim_results} = [@json_files];

	$self->{log}{traces} = JSON->new->decode(
		$self->tar->get_content('src/apps/DriverEval/DriverLog.json') );
	$self->{setup} = JSON->new->decode( $self->tar->get_content('setup.json') );

	mkdir($tmpdir);

	for my $file (@mim_files) {
		$self->tar->extract_file( $file, "${tmpdir}/${file}" );
	}
}

sub load_cache {
	my ($self) = @_;
	my $tmpdir = $self->{tmpdir};
	my ( $dirname, $basename )
	  = ( $self->{data_file} =~ m{ ^ (.*) / ([^/]+) . tar $ }x );
	my $cachefile = "${dirname}/cache/${basename}.json";

	if ( -e $cachefile ) {
		mkdir($tmpdir);
		write_file( $self->json_name, read_file($cachefile) );
		my $json = JSON->new->decode( read_file($cachefile) );
		if ( $json->{version} != $CACHE_VERSION ) {
			return 0;
		}
		$self->{setup} = $json->{setup};
		return 1;
	}
	return 0;
}

sub save_cache {
	my ($self) = @_;
	my $tmpdir = $self->{tmpdir};
	my ( $dirname, $basename )
	  = ( $self->{data_file} =~ m{ ^ (.*) / ([^/]+) . tar $ }x );
	my $cachefile = "${dirname}/cache/${basename}.json";

	if ( not -d "${dirname}/cache" ) {
		mkdir("${dirname}/cache");
	}

	write_file( $cachefile, read_file( $self->json_name ) );
}

sub num_iterations {
	my ($self) = @_;

	return scalar @{ $self->{mim_files} };
}

sub sched_trigger_count {
	my ($self) = @_;

	if ( not $self->{sched_trigger_count} ) {
		$self->{sched_trigger_count} = 0;
		for my $run ( @{ $self->{log}{traces} } ) {
			$self->{sched_trigger_count} += @{ $run->{trace} };
		}
	}

	return $self->{sched_trigger_count};
}

sub merge {
	my ( $self, $file ) = @_;

	if ( not -e $file ) {
		return "Does not exist";
	}

	my $data       = JSON->new->decode( read_file($file) );
	my $trig_count = $data->{triggers};
	if ( $self->sched_trigger_count != $trig_count ) {
		return sprintf( 'Expected %d trigger edges, got %d',
			$self->sched_trigger_count, $trig_count );
	}

	#printf("calibration check at: %.f±%.f  %.f±%.f  %.f±%.f\n",
	#	$data->{calibration}{r0_mean},
	#	$data->{calibration}{r0_std},
	#	$data->{calibration}{r2_mean},
	#	$data->{calibration}{r2_std},
	#	$data->{calibration}{r1_mean},
	#	$data->{calibration}{r1_std},
	#);

	# verify that state duration really is < 1.5 * setup{state_duration} and >
	# 0.5 * setup{state_duration}.  otherwise we may have missed a trigger,
	# which wasn't detected earlier because of duplicate triggers elsewhere.
	my $data_idx = 0;
	for my $run ( @{ $self->{log}{traces} } ) {
		my $prev_elem = {name => q{}};
		for my $trace_elem ( @{ $run->{trace} } ) {
			my $log_elem = $data->{trace}[$data_idx];
			if ($log_elem->{isa} eq 'state'
					and $trace_elem->{name} ne 'UNINITIALIZED'
					and $log_elem->{us} > $self->{setup}{state_duration} * 1500
					and $prev_elem->{name} ne 'txDone'
					and $prev_elem->{name} ne 'epilogue') {
				return sprintf('State %s (trigger index %d) took %.1f ms longer than expected',
					$trace_elem->{name}, $data_idx,
					($log_elem->{us} / 1000) - $self->{setup}{state_duration}
				);
			}
			if ($log_elem->{isa} eq 'state'
					and $trace_elem->{name} ne 'UNINITIALIZED'
					and $trace_elem->{name} ne 'TX'
					and $log_elem->{us} < $self->{setup}{state_duration} * 500 ) {
				return sprintf('State %s (trigger index %d) was %.1f ms shorter than expected',
					$trace_elem->{name}, $data_idx,
					$self->{setup}{state_duration} - ($log_elem->{us} / 1000)
				);
			}
			$prev_elem = $trace_elem;
			$data_idx++;
		}
	}

	$data_idx = 0;
	for my $run ( @{ $self->{log}{traces} } ) {
		for my $trace_elem ( @{ $run->{trace} } ) {
			if ( $data->{trace}[$data_idx]{isa} ne $trace_elem->{isa} ) {
				croak();
			}
			delete $data->{trace}[$data_idx]{isa};
			push( @{ $trace_elem->{offline} }, $data->{trace}[$data_idx] );
			$data_idx++;
		}
	}

	push( @{ $self->{log}{calibration} }, $data->{calibration} );

	return;
}

sub preprocess {
	my ($self)  = @_;
	my $tmpdir  = $self->{tmpdir};
	my @files   = @{ $self->{mim_files} };
	my $shunt   = $self->{setup}{mimosa_shunt};
	my $voltage = $self->{setup}{mimosa_voltage};
	my @errmap;

	@files = map { "${tmpdir}/$_" } @files;

	if ( qx{parallel --version 2> /dev/null} =~ m{GNU parallel} ) {
		system( qw(parallel ../dfatool/bin/analyze.py),
			$voltage, $shunt, ':::', @files );
	}
	else {
		system( qw(parallel ../dfatool/bin/analyze.py),
			$voltage, $shunt, '--', @files );
	}

	for my $i ( 0 .. $#{ $self->{mim_results} } ) {
		my $file = $self->{mim_results}[$i];
		my $error = $self->merge("${tmpdir}/${file}");

		if ($error) {
			say "${file}: ${error}";
			push(@errmap, $i);
		}
	}

	if ( @errmap == @files ) {
		die("All MIMOSA measurements were erroneous. Aborting.\n");
	}

	$self->{log}{model}   = $self->{model};
	$self->{log}{errmap}  = \@errmap;
	$self->{log}{setup}   = $self->{setup};
	$self->{log}{version} = $CACHE_VERSION;
	write_file( $self->json_name,
		JSON->new->convert_blessed->encode( $self->{log} ) );
}

sub analyze {
	my ( $self, @extra_files ) = @_;
	my $tmpdir = $self->{tmpdir};

	@extra_files = grep { $_ ne $self->json_name } @extra_files;

	for my $file ( $self->json_name, @extra_files ) {
		my $json = JSON->new->decode( read_file($file) );
		$json->{model} = $self->{model};

# fix for incomplete json files: transitions can also depend on global parameters
		for my $run ( @{ $json->{traces} } ) {
			for my $i ( 0 .. $#{ $run->{trace} } ) {
				$run->{trace}[$i]{parameter}
				  //= $run->{trace}[ $i - 1 ]{parameter};
			}
		}

		write_file( $file, JSON->new->convert_blessed->encode($json) );
	}

	system( '../dfatool/bin/merge.py', @{ $self->{merge_args} // [] },
		$self->json_name, @extra_files );

	my $json = JSON->new->decode( read_file( $self->json_name ) );

	$self->{aggregate} = $json->{aggregate};

	# debug
	write_file( "/tmp/DriverLog.json", JSON->new->pretty->encode($json) );
}

sub validate {
	my ( $self, @extra_files ) = @_;
	my $tmpdir = $self->{tmpdir};

	@extra_files = grep { $_ ne $self->json_name } @extra_files;

	for my $file ( $self->json_name, @extra_files ) {
		my $json = JSON->new->decode( read_file($file) );
		$json->{model} = $self->{model};
		my @errmap = @{ $json->{errmap} // [] };

# fix for incomplete json files: transitions can also depend on global parameters
		for my $run ( @{ $json->{traces} } ) {
			for my $i ( 0 .. $#{ $run->{trace} } ) {
				$run->{trace}[$i]{parameter}
				  //= $run->{trace}[ $i - 1 ]{parameter};
			}
		}
		# online durations count current state + next transition, but we
		# only want to analyze current state -> substract next transition.
		# Note that we can only do this on online data which has
		# corresponding offline data, i.e. where the offline data was not
		# erroneous
		for my $run ( @{ $json->{traces} } ) {
			if (exists $run->{total_energy}) {
				# splice changes the array (and thus the indices). so we need to
				# start removing elements at the end
				for my $erridx (reverse @errmap) {
					splice(@{$run->{total_energy}}, $erridx, 1);
				}
			}
			for my $i ( 0 .. $#{ $run->{trace} } ) {
				for my $erridx (reverse @errmap) {
					splice(@{$run->{trace}[$i]{online}}, $erridx, 1);
				}
				if ($run->{trace}[$i]{isa} eq 'state') {
					for my $j (0 .. $#{ $run->{trace}[$i]{online} } ) {
						$run->{trace}[$i]{online}[$j]{time} -=
							$run->{trace}[$i+1]{offline}[$j]{us};
					}
				}
			}
		}

		write_file( $file, JSON->new->convert_blessed->encode($json) );
	}

	system( '../dfatool/bin/merge.py', @{ $self->{merge_args} // [] },
		'--validate', $self->json_name, @extra_files );

	my $json = JSON->new->decode( read_file( $self->json_name ) );

	$self->{aggregate} = $json->{aggregate};

	# debug
	write_file( "/tmp/DriverLog.json", JSON->new->pretty->encode($json) );
}

sub crossvalidate {
	my ( $self, @extra_files ) = @_;
	my $tmpdir = $self->{tmpdir};

	@extra_files = grep { $_ ne $self->json_name } @extra_files;

	for my $file ( $self->json_name, @extra_files ) {
		my $json = JSON->new->decode( read_file($file) );
		$json->{model} = $self->{model};

# fix for incomplete json files: transitions can also depend on global parameters
		for my $run ( @{ $json->{traces} } ) {
			for my $i ( 0 .. $#{ $run->{trace} } ) {
				$run->{trace}[$i]{parameter}
				  //= $run->{trace}[ $i - 1 ]{parameter};
			}
		}

		write_file( $file, JSON->new->convert_blessed->encode($json) );
	}

	system( '../dfatool/bin/merge.py', @{ $self->{merge_args} // [] },
		'--crossvalidate', $self->json_name, @extra_files );
}

sub data {
	my ($self) = @_;
	my $tmpdir = $self->{tmpdir};
	my $json   = JSON->new->decode( read_file( $self->json_name ) );
	return $json;
}

sub json_name {
	my ($self) = @_;
	my $tmpdir = $self->{tmpdir};

	return "${tmpdir}/DriverLog.json";
}

1;
