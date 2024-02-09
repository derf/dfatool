FROM debian:bullseye

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_LISTCHANGES_FRONTEND=none

RUN apt-get update \
	&& apt-get -y --no-install-recommends install \
		ca-certificates \
		kconfig-frontends-nox \
		make \
		python3-dev \
		python3-coverage \
		python3-matplotlib \
		python3-numpy \
		python3-pytest \
		python3-pytest-cov \
		python3-scipy \
		python3-sklearn \
		python3-yaml \
		python3-zbar \
		wget \
	&& rm -rf /var/cache/apt/* /var/lib/apt/lists/*

CMD ["bash"]
