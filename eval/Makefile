FILES = acher2022linux1-cart
FILES += acher2022linux2-cart
FILES += busybox-cart busybox-nmt busybox-rmt
FILES += reskil-cart reskil-nmt reskil-rmt
FILES += siegmund2015dune-cart siegmund2015dune-nmt siegmund2015dune-rmt
FILES += siegmund2015hipacc-cart siegmund2015hipacc-nmt siegmund2015hipacc-rmt
FILES += siegmund2015hsmgp-cart siegmund2015hsmgp-nmt siegmund2015hsmgp-rmt
FILES += siegmund2015javagc-cart siegmund2015javagc-nmt siegmund2015javagc-rmt
FILES += siegmund2015sac-cart siegmund2015sac-nmt siegmund2015sac-rmt
FILES += siegmund2015x264-cart siegmund2015x264-nmt siegmund2015x264-rmt
FILES += x264-cart x264-nmt x264-rmt

default: $(addprefix out/, $(addsuffix .tex, ${FILES}))

out/%.tex: scripts/%.sh ../.git/index
	scripts/$*.sh
