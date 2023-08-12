#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <omp.h>
#include <stdint.h>

#define MOD4_MAX_LEN	64
#define MAX_NUMBERS		1000
#define MAX_BYTES		24
#define MAX_PRIMES		168
#define THREADS_LEVELS	2
#define THREADS_LENS	2
#define STEP			1
#define START_NUMBER	1

typedef struct
{
	int32_t n;
	int32_t set_len;
	int32_t set_len_minus_one;
	int16_t buffer[MOD4_MAX_LEN][MOD4_MAX_LEN + 8];

	int64_t r4, qop, rank;
}param_t;

typedef struct
{
	int64_t r4, qop, rank;
}value_t;

int32_t g_primes[MAX_PRIMES];
uint8_t g_decompos[MAX_NUMBERS][MAX_BYTES];

static void gen_primes(void)
{
	int32_t i, j, k, nums[MAX_NUMBERS];

	for (i = 0; i < MAX_NUMBERS; i++)
		nums[i] = i;

	for (k = 0, i = 2; i < MAX_NUMBERS; i++)
	{
		if (nums[i] == 0)
			continue;

		g_primes[k++] = nums[i];
		for (j = i * i; j < MAX_NUMBERS; j += i)
			nums[j] = 0;
	}
}

static void gen_decompos(void)
{
	uint8_t* bitset;
	int32_t i, j, tmp;

	for (i = 2; i < MAX_NUMBERS; i++)
	{
		for (tmp = i, j = 0; j < MAX_PRIMES; j++)
		{
			while (tmp % g_primes[j] == 0)
			{
				bitset = &g_decompos[i][j >> 3];
				if (*bitset & (1 << (j & 7)))
					*bitset &= ~(1 << (j & 7));
				else
					*bitset |= (1 << (j & 7));
				tmp /= g_primes[j];
			}
		}
	}
}

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
				int16_t* cur;
				int64_t* src, * dst;

				dst = (int64_t*)param->buffer[step];
				src = (int64_t*)param->buffer[step - 1];
				while (*src) *dst++ = *src++;

				lookup_r(step + 1, sum, param);
				sum += (param->set_len - step) << 1;

				cur = &param->buffer[step - 1][step];
				while (*cur) *cur++ += 2;
			}
			else
			{
				param->buffer[step - 1][param->set_len_minus_one] += param->n - sum;
				sum = param->n;
			}
		}
		else if (sum == param->n)
		{
			int16_t* cur;
			int64_t* dec;
			int64_t factors[3] = { 0 };

			cur = param->buffer[step - 1];
			while (*cur)
			{
				dec = (int64_t*)g_decompos[*cur++];
				factors[0] = factors[0] ^ dec[0];
				factors[1] = factors[1] ^ dec[1];
				factors[2] = factors[2] ^ dec[2];
			}

			param->r4++;
			if (factors[0] || factors[1] || factors[2])
				param->rank++;
			else
				param->qop++;
			return;
		}
		else
			return;
	}
}

static void process_len(int32_t n, int32_t set_len, value_t* ret)
{
	int64_t r4, qop, rank;
	int32_t i, num_levels;

	if (set_len == 1)
	{
		ret->r4 = 0;
		ret->qop = 0;
		ret->rank = 0;

		if (n % 2 != 0)
		{
			double a = sqrt((double)n);

			ret->r4 = 1;
			if (a == (int64_t)a)
				ret->qop = 1;
			else
				ret->rank = 1;
		}
		return;
	}

	qop = r4 = rank = 0;
	num_levels = levels_count(n, set_len);

	#pragma omp parallel for num_threads(THREADS_LEVELS) schedule(dynamic, 1) reduction(+:r4, qop, rank)
	for (i = 0; i < num_levels; i++)
	{
		int32_t m, sum;
		param_t param = { 0 };

		param.n = n;
		param.set_len = set_len;
		param.set_len_minus_one = param.set_len - 1;

		sum = 0;
		for (m = 0; m < param.set_len; m++)
		{
			param.buffer[0][m] = (1 + (i << 1)) + (m << 1);
			sum += param.buffer[0][m];
		}

		lookup_r(1, sum, &param);
		r4 += param.r4;
		qop += param.qop;
		rank += param.rank;
	}

	ret->r4 = r4;
	ret->qop = qop;
	ret->rank = rank;
}

static void process_number(int32_t n)
{
	double start;
	int64_t r4, qop, rank;
	int32_t i, lens[MOD4_MAX_LEN], count;

	qop = r4 = rank = 0;
	count = set_len_fill(n, lens);
	start = omp_get_wtime();

	#pragma omp parallel for num_threads(THREADS_LENS) schedule(dynamic, 1) reduction(+:r4, qop, rank)
	for (i = 0; i < count; i++)
	{
		value_t ret;

		process_len(n, lens[i], &ret);
		r4 += ret.r4;
		qop += ret.qop;
		rank += ret.rank;
	}

	printf("%d) r4 = %llu, qop = %llu, rank = %llu, %.3f\n", n, r4, qop, rank, omp_get_wtime() - start);
}

int main(int argc, char* argv[])
{
	int32_t n;

	gen_primes();
	gen_decompos();

	omp_set_nested(1);
	for (n = START_NUMBER; n <= MAX_NUMBERS; n += STEP)
		process_number(n);

	return 0;
}