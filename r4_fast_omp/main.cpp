#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define MOD4_MAX_LEN	64
#define MAX_NUMBERS		1000
#define START_NUMBER	1
#define STEP			1
#define THREADS_LEVELS	2
#define THREADS_LENS	2

typedef struct
{
	int32_t n;
	int32_t set_len;
	int32_t set_len_minus_one;

	int64_t r4;
}param_t;

static int32_t levels_count(int32_t n, int32_t set_len)
{
	int32_t levels, sum, i;

	levels = sum = 0;
	for (i = 0; i < set_len; i++)
		sum += 1 + (i << 1);

	while (1)
	{
		if (sum > n)
			return levels;
		else if (sum == n)
			return ++levels;
		else
		{
			levels++;
			sum += (set_len << 1);
		}
	}
}

static int32_t set_len_fill(int32_t n, int32_t* lens)
{
	int32_t elemnum, eq, i, count;

	elemnum = (int32_t)sqrt((float)n);
	if (elemnum > MOD4_MAX_LEN)
		elemnum = MOD4_MAX_LEN;

	eq = n % 4;
	count = 0;
	for (i = 1; i <= elemnum; i++)
	{
		if (eq == (i % 4))
			lens[count++] = i;
	}

	return count;
}

static void lookup_r(int32_t step, int32_t sum, param_t* param)
{
	while (1)
	{
		if (sum < param->n)
		{
			if (param->set_len_minus_one != step)
			{
				lookup_r(step + 1, sum, param);
				sum += (param->set_len - step) << 1;
			}
			else
			{
				param->r4++;
				return;
			}
		}
		else if (sum == param->n)
		{
			param->r4++;
			return;
		}
		else
			return;
	}
}

static int64_t process_len(int32_t n, int32_t set_len)
{
	int64_t r4;
	int32_t i, num_levels;

	if (set_len == 1)
	{
		if (n % 2 != 0)
			return 1;
		return 0;
	}

	r4 = 0;
	num_levels = levels_count(n, set_len);
	#pragma omp parallel for num_threads(THREADS_LEVELS) schedule(dynamic, 1) reduction(+:r4)
	for (i = 0; i < num_levels; i++)
	{
		int32_t m, sum;
		param_t param;

		param.n = n;
		param.r4 = 0;
		param.set_len = set_len;
		param.set_len_minus_one = param.set_len - 1;

		sum = 0;
		for (m = 0; m < param.set_len; m++)
			sum += (1 + (i << 1)) + (m << 1);

		lookup_r(1, sum, &param);
		r4 += param.r4;
	}

	return r4;
}

static void process_number(int32_t n)
{
	int64_t r4;
	double start;
	int32_t i, lens[MOD4_MAX_LEN], count;

	r4 = 0;
	count = set_len_fill(n, lens);
	start = omp_get_wtime();

	#pragma omp parallel for num_threads(THREADS_LENS) schedule(dynamic, 1) reduction(+:r4)
	for (i = 0; i < count; i++)
		r4 += process_len(n, lens[i]);

	printf("%d) r4 = %llu, %.3f\n", n, r4, omp_get_wtime() - start);
}

int main(int argc, char* argv[])
{
	int32_t n;

	omp_set_nested(1);
	for (n = START_NUMBER; n <= MAX_NUMBERS; n += STEP)
		process_number(n);

	return 0;
}