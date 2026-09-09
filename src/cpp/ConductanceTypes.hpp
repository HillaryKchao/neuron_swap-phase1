#pragma once

// Host-side conductance types.
// Use this header for ordinary C++ simulation, waveform generation, and any
// workflow that still relies on the standard library.

#include <algorithm>
#include <cmath>
#include <random>

using scalar_t = double;

struct StochKVTransitions {
	int n0_to_n1;
	int n1_to_n0;
};

inline StochKVTransitions StochKV_sample_transitions(scalar_t n, scalar_t V, scalar_t celsius, scalar_t area, scalar_t dt, std::mt19937& rng) {
	constexpr scalar_t THA = -40.0;
	constexpr scalar_t QA = 9.0;
	constexpr scalar_t RA = 0.02;
	constexpr scalar_t RB = 0.002;
	constexpr scalar_t Q10 = 2.3;
	constexpr scalar_t TEMP = 23.0;
	constexpr scalar_t e = 2.718281828459045235360287471352;
	constexpr scalar_t gK_bar = 0.75;
	constexpr scalar_t gamma = 30.0;

	const scalar_t tadj = std::pow(Q10, ((celsius - TEMP) / 10.0));

	scalar_t a = 0.0;
	scalar_t b = 0.0;
	if (std::fabs(V - THA) > 1e-6) {
		a = RA * (V - THA) / (1.0 - std::pow(e, ((THA - V) / QA)));
		b = -RB * (V - THA) / (1.0 - std::pow(e, ((THA - V) / QA)));
	} else {
		a = RA * QA;
		b = RB * QA;
	}
	a *= tadj;
	b *= tadj;

	const scalar_t eta = gK_bar / gamma;
	const int N = static_cast<int>(std::floor(eta * area + 0.5));
	const int N1 = static_cast<int>(std::floor(n * N + 0.5));
	const int N0 = N - N1;

	const scalar_t P_a = std::max<scalar_t>(0.0, std::min<scalar_t>(1.0, a * dt));
	const scalar_t P_b = std::max<scalar_t>(0.0, std::min<scalar_t>(1.0, b * dt));

	std::binomial_distribution<int> dist_a(N0, P_a);
	std::binomial_distribution<int> dist_b(N1, P_b);
	return {dist_a(rng), dist_b(rng)};
}