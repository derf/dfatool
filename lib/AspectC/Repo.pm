package AspectC::Repo;

use strict;
use warnings;
use 5.020;
use List::Util qw(first);
use XML::LibXML;

our $VERSION = '0.00';

my @source_loc_kind = (qw(none definition declaration));
my @function_kind   = (
	qw(unknown non_member static_non_member member
	  static_member virtual_member pure_virtual_member conctructor destructor
	  virtual_destructor pure_virtual_destructor)
);
my @pointcut_kind = (qw(normal virtual pure_virtual));
my @variable_kind
  = (qw(unknown non_member static_non_member member static_member));
my @advice_code_kind = (qw(before after around));
my @advice_code_context
  = (qw(none type obj type_obj vars type_vars obj_vars type_obj_vars));
my @cv_qualifiers = (qw(none const volatile const_volatile));

sub new {
	my ( $class, %opt ) = @_;

	my $self = \%opt;

	$self->{xml}
	  = XML::LibXML->load_xml( location => '../kratos/src/repo.acp' );

	bless( $self, $class );
	$self->parse_xml;
	return $self;
}

sub parse_xml {
	my ($self) = @_;

	my $xml = $self->{xml};

	for my $node (
		$xml->findnodes('/ac-model/files/TUnit | /ac-model/files/Header') )
	{
		my $filename = $node->getAttribute('filename');
		my $id       = $node->getAttribute('id');
		if ( defined $id ) {
			$self->{files}{$id} = $filename;
		}
		else {
			say STDERR "repo.acp: File ${filename} has no ID";
		}
	}

	for my $aspect (
		$xml->findnodes(
			    '/ac-model/root/Namespace[@name="::"]/children/Aspect'
			  . ' | /ac-model/root/Namespace[@name="::"]/children/Namespace'
		)
	  )
	{
		my $aspect_name = $aspect->getAttribute('name');
		for my $attr_node ( $aspect->findnodes('./children/Attribute') ) {
			my $attr_name = $attr_node->getAttribute('name');
			my $attr_id   = $attr_node->getAttribute('id');
			my @args;
			for my $arg_node ( $attr_node->findnodes('./args/Arg') ) {
				my $arg_name = $arg_node->getAttribute('name');
				my $arg_type = $arg_node->getAttribute('type');
				push(
					@args,
					{
						name => $arg_name,
						type => $arg_type,
					}
				);
			}
			if ( defined $attr_id ) {
				$self->{attributes}{$attr_id} = {
					namespace => $aspect_name,
					name      => $attr_name,
					arguments => \@args,
				};
			}
		}
	}

	for my $node (
		$xml->findnodes('/ac-model/root/Namespace[@name="::"]/children/Class') )
	{
		my $class      = {};
		my $class_name = $node->getAttribute('name');
		my $id         = $node->getAttribute('id');
		my @bases;
		my @functions;
		my @sources;
		if ( my $base_str = $node->getAttribute('bases') ) {
			@bases = split( qr{ }, $base_str );
		}

		for my $source ( $node->findnodes('./source/Source') ) {
			push(
				@sources,
				{
					file => $self->{files}{ $source->getAttribute('file') },
					kind => $source_loc_kind[ $source->getAttribute('kind') ],
				}
			);
		}

		$class->{name}    = $class_name;
		$class->{id}      = $id;
		$class->{sources} = [@sources];

		for my $fnode ( $node->findnodes('./children/Function') ) {
			my $name        = $fnode->getAttribute('name');
			my $id          = $fnode->getAttribute('id') // q{?};
			my $kind        = $fnode->getAttribute('kind');
			my $result_type = q{?};
			my @args;

			if ( my $typenode = ( $fnode->findnodes('./result_type/Type') )[0] )
			{
				$result_type = $typenode->getAttribute('signature');
			}

			for my $argnode ( $fnode->findnodes('./arg_types/Type') ) {
				push( @args, $argnode->getAttribute('signature') );
			}

			my $fun = {
				name     => $name,
				returns  => $result_type,
				argtypes => [@args],
			};

			for my $annotation_node (
				$fnode->findnodes('./annotations/Annotation') )
			{
				my $attr_id   = $annotation_node->getAttribute('attribute');
				my $attribute = {
					name      => $self->{attributes}{$attr_id}{name},
					namespace => $self->{attributes}{$attr_id}{namespace},
				};
				for my $param_node (
					$annotation_node->findnodes('./parameters/Parameter') )
				{
					my $value      = $param_node->getAttribute('value');
					my $expression = $param_node->getAttribute('expression');
					push(
						@{ $attribute->{args} },
						{
							value      => $value,
							expression => $expression,
						}
					);
				}
				push( @{ $fun->{attributes} }, $attribute );
			}

			my $hash_key = sprintf( '%s(%s)', $name, join( q{, }, @args ) );
			$class->{function}{$hash_key} = $fun;
		}

		for my $vnode ( $node->findnodes('./children/Variable') ) {
			my $name       = $vnode->getAttribute('name');
			my ($sig_node) = $vnode->findnodes('./type/Type');
			my $signature  = $vnode->getAttribute('signature');
			$class->{variable}{$name} = {
				type => $signature,
			};
		}

		$self->{class}{$class_name} = $class;
	}

	for my $node (
		$xml->findnodes(
			'/ac-model/root/Namespace[@name="::"]/children/Variable')
	  )
	{
		my $sig_node  = ( $node->findnodes('./type/Type') )[0];
		my $kind      = $node->getAttribute('kind');
		my $name      = $node->getAttribute('name');
		my $signature = $sig_node->getAttribute('signature');

		if ( $variable_kind[$kind] eq 'non_member' ) {
			$self->{class_instance}{$signature} = $name;
		}
	}

	return $self;
}

sub get_class_path_prefix {
	my ( $self, $class_name ) = @_;

	my $header = first { $_->{kind} eq 'definition' }
	@{ $self->{class}{$class_name}{sources} };
	$header = $header->{file};
	$header =~ s{ \. h $ }{}x;

	return $header;
}

sub get_class_instance {
	my ( $self, $class_name ) = @_;

	return $self->{class_instance}{$class_name};
}

1;
