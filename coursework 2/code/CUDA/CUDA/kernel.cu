//defines the softening to make sure inf do not occur
#define SOFTENING 2E4
#define _USE_MATH_DEFINES
//number of particles in simulation
#define nParticles 4000
// time step
#define dt 1.0f;
 

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

//particle struct
struct Particle
{
	double x, y;       // positions
	double vx, vy;     // velocity 
	double fx, fy;     // force 
	double mass;
	ALLEGRO_COLOR colour;

};
//creates array of particles 
Particle p[nParticles];
//creates an instance of a particle for cuda to use
Particle *singlep;
const int nIters = 10;  // simulation iterations
//defines the block for CUDA
int Block = 16;
//created a long to allocate memory size for cuda
long massive;
//function to create the particles
void CreateParticles()
{
	//for loop that iterates through all the particles
	for (int i = 0; i < nParticles; i++)
	{
		// assigns a random location for each particle 
		p[i].x = rand() % 5000 - 2500;
		p[i].y = rand() % 5000 - 2500; 
		//sets the speed of the particle to 0
		p[i].vx = 0.0f;
		p[i].vy = 0.0f;
		//sets the mass to a random size
		p[i].mass = rand() % 1000 + 1;
		//randomizes the colour of the particle
		int red = rand() % 255;
		int blue = rand() % 255;
		int green = rand() % 255;
		p[i].colour = al_map_rgb(red, blue, green);	
	}
	//Copies count massive from the memory area pointed to by p to the memory area pointed to by singlep
	cudaMemcpy(singlep, &p, massive, cudaMemcpyHostToDevice);
}
//funtion that draws the particles to the screen
void DrawParticles()
{
	for (int i = 0; i < nParticles; i++)
	{
		p[i].x += p[i].vx*dt;
		p[i].y += p[i].vy*dt;
		//uses allegro fuction to draw the particles to the middle of the screen
		al_draw_filled_circle((800 / 2) + ((int)p[i].x) / 10, (600 / 2) + ((int)p[i].y) / 10, p[i].mass/500, p[i].colour);
	}
}
//initialize funtion
void Init()
{
	//makes sure random is not the same every time the programme is run
	srand(time(NULL));
	//initializes allegro 
	al_init();
	//this function allows 
	al_init_primitives_addon();
	//creates 
	ALLEGRO_DISPLAY* display = al_create_display(800, 600);
	cudaSetDevice(0);
	massive = sizeof(Particle) * nParticles;
	//allocates the memory needed for cuda 
	cudaMalloc((void**)&singlep, massive);
}
//function that is to be parralezied
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
	singlep[i].vy += Fy * dt =;
	__syncthreads();
}
void runSim(int b)
{
	int number_of_blocks = (nParticles + Block - 1) / Block;
	cudaMemcpy(singlep, &p, massive, cudaMemcpyHostToDevice);
	ParticleForce << < number_of_blocks, Block >> > (singlep);
	cudaDeviceSynchronize();
	cudaMemcpy(&p, &singlep[0], massive, cudaMemcpyDeviceToHost);
	DrawParticles();
	al_flip_display();
	al_clear_to_color(al_map_rgb(0, 0, 0));
}

int main(const int argc, const char** argv)
{
	Init();
	//starts the timing device
	auto start = std::chrono::system_clock::now();
	int b = 16;
	//runs through the programm 10 times to calculate the average
	for (int iter = 0; iter <= nIters; iter++)
	{
		CreateParticles();
		auto current_start = std::chrono::system_clock::now();
		//runs through the simulation many times
		for (int i = 0; i < 10000; i++)
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
	//allows the programme to 
	int a;
	std::cin >> a;
	al_clear_to_color(al_map_rgb(0, 0, 0));
}

