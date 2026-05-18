#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>
#include <unistd.h>

int main(void)
{
	printf("x y\n");
	for (int i = 1; i <= 100; i++) {
		unsigned long long t0 = __rdtsc();
		sleep(1);
		unsigned long long t1 = __rdtsc();
		printf("%d %llu\n", i, t1 - t0);
	}
	return 0;
}
