# docker build -t dfatool:latest .

FROM debian:buster-slim as files

COPY bin/ /dfatool/bin/
COPY lib/ /dfatool/lib/
COPY ext/ /dfatool/ext/

RUN sed -i 's/charset is "C"/charset == "C"/' /dfatool/bin/dfatool/pubcode/code128.py

FROM debian:bullseye

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_LISTCHANGES_FRONTEND=none

WORKDIR /dfatool

RUN apt-get update \
	&& apt-get -y --no-install-recommends install \
		ca-certificates \
		python3-dev \
		python3-coverage \
		python3-matplotlib \
		python3-numpy \
		python3-pytest \
		python3-pytest-cov \
		python3-scipy \
		python3-sklearn \
		python3-ubjson \
		python3-yaml \
		python3-zbar \
	&& rm -rf /var/cache/apt/* /var/lib/apt/lists/*

COPY --from=files /dfatool/ /dfatool/
