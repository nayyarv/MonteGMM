#include <stdlib.h>
#include <math.h>

float gaussrand()
{
	static float V1, V2, S;
	static int phase = 0;
	float X;

	if(phase == 0) {
		do {
			float U1 = rand() / RAND_MAX;
			float U2 = rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}