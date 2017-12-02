#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <omp.h>
#include <iostream>
#include "Particle.h"
#include <chrono>
#include <allegro5\allegro.h>
#include <allegro5\allegro_font.h>
#include <allegro5\allegro_primitives.h>
#include <thread>

#define SOFTENING 2E4



vector<Particle*> p;
const float dt = 0.1f;// time step
int nParticles = 0;
const int nIters = 10;  // simulation iterations

int bytes = nParticles * sizeof(p);
float *buf = (float*)malloc(bytes);

void CreateParticles(int nParticles)
{
	for (int i = 0; i < nParticles; i++)
	{
		float x = rand() % 5000 - 2500;
		float y = rand() % 5000 - 2500;
		float vx = 0.0f, vy = 0.0f;
		//// Orient a random 2D circular orbit
		if (rand() <= .5)
		{
			vx = -vx;
			vy = -vy;
		}
		//double mass = rand() / SOFTENING * 10 + 1e20;
		double mass = rand() % 1000 + 1;
		//// Color the masses in green gradients by mass
		int red = rand() % 255;
		int blue = rand() % 255;
		int green = rand() % 255;
		ALLEGRO_COLOR colour = al_map_rgb(red, blue, green);
		//// put a heavy body in the center
		if (i == 0)
		{
			p.push_back(new Particle(0.1, 0.5, 0.0f, 0.0f, mass, colour));
		}
		p.push_back(new Particle(x, y, vx, vy, mass, colour));
	}
}

void ParticleForce(vector<Particle*> p, float dt, int n, int b)
{ 
	for (int i = 0; i < n; i++)
	{
		float Fx = 0.0f, Fy = 0.0f;
		for (int j = 0; j < n; j++)
		{
			float dx = p[j]->x - p[i]->x;
			float dy = p[j]->y - p[i]->y;
			float distSqr = dx*dx + dy*dy;
			//if (distSqr == 0)
			//	distSqr = 0.01;
			//if (distSqr != 0.0f)
			//{
			float invDist = 1.0f / sqrtf(distSqr + SOFTENING);
			float invDist3 = invDist * invDist * invDist;
			Fx += (p[i]->mass) * dx * invDist3;
			Fy += /*(p[i]->mass )*/  dy * invDist3;
			//}
		}
		p[i]->vx += dt* Fx;
		p[i]->vy += dt* Fy;
	}
	for (int i = 0; i < n; i++)
	{
		p[i]->x += p[i]->vx*dt * b;
		p[i]->y += p[i]->vy*dt *b;
	}
}

void DrawParticles()
{
	for (int i = 0; i < nParticles; i++)
	{
		al_draw_filled_circle((800 / 2) + ((int)p[i]->x) / 10, (600 / 2) + ((int)p[i]->y) / 10, 1.75f, p[i]->colour);
	}
}

int main(const int argc, const char** argv)
{
	srand(time(NULL));

	al_init();
	al_init_primitives_addon();
	ALLEGRO_DISPLAY* display = al_create_display(800, 600);

	auto start = std::chrono::system_clock::now();
	int b = 4000;
	for (int j = 0; j <= 6; j++)
	{
		nParticles += b;
		for (int iter = 0; iter <= nIters; iter++)
		{
			p.clear();
			CreateParticles(nParticles);
			auto current_start = std::chrono::system_clock::now();
			for (int i = 0; i < 1000; i++)
			{
				ParticleForce(p, dt, nParticles, b);
				DrawParticles();
				al_flip_display();
				al_clear_to_color(al_map_rgb(0, 0, 0));

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
	}
	int a;
	std::cin >> a;
	al_clear_to_color(al_map_rgb(0, 0, 0));
}


