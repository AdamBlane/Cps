#pragma once
using namespace std;
#include <math.h>
#include <allegro5/allegro_color.h>
class Particle
{
	//private:
	//	const double G = 6.673e-11;   // gravitational constant
	//	const double solarmass = 1.98892e30;
	public:
		double x, y;       // positions
		double vx, vy;     // velocity 
		double fx, fy;     // force 
		double mass;        
		ALLEGRO_COLOR colour;

		Particle();
		~Particle();	
		Particle(double x, double y, double vx, double vy, double mass, ALLEGRO_COLOR color);
		//for the update function
		void Update(float delta_time);

};