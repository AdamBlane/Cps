#define SOFTENING 2E4
#define _USE_MATH_DEFINES
#define nParticles 3250
#define dt 0.1f;// time step

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <allegro5\allegro.h>
#include <allegro5\allegro_primitives.h>
#include <curand.h>
#include <curand_kernel.h>

struct Particle
{
	double x, y;       // positions
	double vx, vy;     // velocity 
	double fx, fy;     // force 
	double mass;
	ALLEGRO_COLOR colour;

};

Particle p[nParticles];
Particle *singlep;
const int nIters = 10;  // simulation iterations
int Block = 16;
long bytes;
float *buf = (float*)malloc(bytes);

void CreateParticles()
{
	for (int i = 0; i < nParticles; i++)
	{
		float x = rand() % 5000 - 2500;
		float y = rand() % 5000 - 2500;
		float vx = 0.0f, vy = 0.0f;
		if (rand() <= .5)
		{
			vx = -vx;
			vy = -vy;
		}
		//double mass = rand() / SOFTENING * 10 + 1e20;
		double mass = rand() % 1000 + 1;
		int red = rand() % 255;
		int blue = rand() % 255;
		int green = rand() % 255;
		ALLEGRO_COLOR colour = al_map_rgb(red, blue, green);
		if (i == 0)
		{
			p[i].x = 0.1;
			p[i].y = 0.5; 
			p[i].vx = 0.0f;
			p[i].vy = 0.0f;
			p[i].mass = mass; 
			p[i].colour = colour;
		}
		p[i].x = x;
		p[i].y = y;
		p[i].vx = vx;
		p[i].vy = vy;
		p[i].mass = mass;
		p[i].colour = colour;
	}
	cudaMemcpy(singlep, &p, nParticles, cudaMemcpyHostToDevice);
}

void DrawParticles()
{
	for (int i = 0; i < nParticles; i++)
	{
		p[i].x += p[i].vx*dt;
		p[i].y += p[i].vy*dt;
		al_draw_filled_circle((800 / 2) + ((int)p[i].x) / 10, (600 / 2) + ((int)p[i].y) / 10, 1.75f, p[i].colour);
	}
}
void Init()
{
	srand(time(NULL));
	al_init();
	al_init_primitives_addon();
	ALLEGRO_DISPLAY* display = al_create_display(800, 600);

	cudaSetDevice(0);
	bytes = sizeof(p) * nParticles;
	cudaMalloc((void**)&singlep, bytes);
}
__global__
void ParticleForce(Particle* singlep)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float Fx = 0.0f, Fy = 0.0f;
	for (int j = 0; j < nParticles; j++)
	{
		float dx = singlep[j].x - singlep[i].x;
		float dy = singlep[j].y - singlep[i].y;
		float distSqr = dx*dx + dy*dy;
		float invDist = 1.0f / sqrtf(distSqr + SOFTENING);
		float invDist3 = invDist * invDist * invDist;
		Fx += (singlep[i].mass) * dx * invDist3;
		Fy += /*(p[i]->mass )*/  dy * invDist3;
	}
	singlep[i].vx += Fx * dt;
	singlep[i].vy += Fy * dt ;
	__syncthreads();
}
void runSim(int b)
{
	int number_of_blocks = (nParticles + Block - 1) / Block;
	cudaMemcpy(singlep, &p, nParticles , cudaMemcpyHostToDevice);
	ParticleForce << < number_of_blocks, Block >> > (singlep);
	cudaDeviceSynchronize();
	cudaMemcpy(&p, &singlep[0], nParticles, cudaMemcpyDeviceToHost);
	DrawParticles();
	al_flip_display();
	al_clear_to_color(al_map_rgb(0, 0, 0));
}

int main(const int argc, const char** argv)
{
	Init();
	auto start = std::chrono::system_clock::now();
	int b = 16;
	for (int iter = 0; iter <= nIters; iter++)
	{
		CreateParticles();
		auto current_start = std::chrono::system_clock::now();
		for (int i = 0; i < 1000; i++)
		{
			runSim(b);
		}
		// Get the end time
		auto current_end = std::chrono::system_clock::now();
		// Get the total time
		auto current_total = current_end - current_start;
		std::cout << "Iteration: " << iter << " Time Taken:  " << std::chrono::duration_cast<std::chrono::milliseconds>(current_total).count() << std::endl;
	}
	// Get the end time
	auto end = std::chrono::system_clock::now();
	// Get the total time
	auto total = end - start;
	int a;
	std::cin >> a;
	al_clear_to_color(al_map_rgb(0, 0, 0));
}

