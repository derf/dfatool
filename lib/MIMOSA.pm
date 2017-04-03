package MIMOSA;

use strict;
use warnings;
use 5.020;

use Carp;
use Carp::Assert::More;
use File::Slurp qw(read_dir);
use IPC::Run qw(harness);
use List::Util qw(max);

our $VERSION = '0.00';

sub new {
	my ( $class, %opt ) = @_;

	my $self = \%opt;

	bless( $self, $class );

	return $self;
}

sub start {
	my ($self) = @_;
	my $buf;

	my $mim_daemon = harness(
		[ 'MimosaCMD', '--start' ],
		'<'  => \undef,
		'>&' => \$buf
	);
	my $mim_p1 = harness(
		[ 'MimosaCMD', '--parameter', 'offset', $self->{mimosa_offset} ],
		'<'  => \undef,
		'>&' => \$buf
	);
	my $mim_p2 = harness(
		[ 'MimosaCMD', '--parameter', 'shunt', $self->{mimosa_shunt} ],
		'<'  => \undef,
		'>&' => \$buf
	);
	my $mim_p3 = harness(
		[ 'MimosaCMD', '--parameter', 'voltage', $self->{mimosa_voltage} ],
		'<'  => \undef,
		'>&' => \$buf
	);
	my $mim_p4 = harness(
		[ 'MimosaCMD', '--parameter', 'directory', 'src/apps/DriverEval' ],
		'<'  => \undef,
		'>&' => \$buf
	);
	my $mim_start = harness(
		[ 'MimosaCMD', '--mimosa-start' ],
		'<'  => \undef,
		'>&' => \$buf
	);

	if ( $self->is_running ) {
		croak("MIMOSA daemon is already running");
	}

	$mim_daemon->run or croak();
	$mim_p1->run     or croak();
	$mim_p2->run     or croak();
	$mim_p3->run     or croak();
	$mim_start->run  or croak();
}

sub is_running {
	my ($self) = @_;

	my $buf;

	my $mim_check = harness(
		[ 'pidof', 'MimosaCMD' ],
		'<'  => \undef,
		'>&' => \$buf
	);

	return $mim_check->run;
}

sub stop {
	my ($self) = @_;
	my $buf;

	my $mim_stop = harness(
		[ 'MimosaCMD', '--mimosa-stop' ],
		'<'  => \undef,
		'>&' => \$buf
	);
	my $mim_kill = harness(
		[ 'MimosaCMD', '--stop' ],
		'<'  => \undef,
		'>&' => \$buf
	);

	# make sure MIMOSA has all teh data
	sleep(5);

	$mim_stop->run or croak();

	$self->wait_for_save;

	$mim_kill->run or croak();

	while ( $self->is_running ) {
		sleep(1);
	}
}

sub wait_for_save {
	my ($self) = @_;

	my $mtime         = 0;
	my $mtime_changed = 1;

	while ($mtime_changed) {
		sleep(3);
		my @mim_files = grep { m{ \. mim $ }x } read_dir('.');
		my @mtimes    = map  { ( stat($_) )[9] } @mim_files;
		my $new_mtime = max @mtimes;
		if ( $new_mtime != $mtime ) {
			$mtime = $new_mtime;
		}
		else {
			$mtime_changed = 0;
		}
	}

	return $self;
}

sub kill {
	my ($self) = @_;
	my $buf;

	my $mim_kill = harness(
		[ 'MimosaCMD', '--stop' ],
		'<'  => \undef,
		'>&' => \$buf
	);

	$mim_kill->run or croak();
}

sub calibrate {
	my ($self) = @_;

	$self->mimosactl('disconnect');
	sleep(2);
	$self->mimosactl('1k');      # actually 987 Ohm
	sleep(2);
	$self->mimosactl('100k');    # actually 99.3 kOhm
	sleep(2);
	$self->mimosactl('connect');
}

sub mimosactl {
	my ( $self, $arg ) = @_;
	my $buf;

	my $mimosactl = harness(
		[ 'mimosactl', $arg ],
		'<'  => \undef,
		'>&' => \$buf
	);

	$mimosactl->run
	  or croak( "mimosactl $arg returned " . $mimosactl->full_result );

	return $self;
}

1;
