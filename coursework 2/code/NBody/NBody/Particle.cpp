#include "Particle.h"

Particle::Particle() 
{

}
Particle::Particle(double valuex, double valuey, double valuevx, double valuevy, double valuemass, ALLEGRO_COLOR valuecolour)
{
	this->x = valuex;
	this->y = valuey;
	this->vx = valuevx;
	this->vy = valuevy;
	this->mass = valuemass;
	this->colour = valuecolour;
}
void Particle::Update(float delta_time)
{
	vx += delta_time * fx / mass;
	vy += delta_time * fy / mass;
	x += delta_time * vx;
	y += delta_time * vy;
}
