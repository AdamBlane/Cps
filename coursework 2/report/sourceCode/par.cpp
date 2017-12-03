for (int i = 0; i < n; i++)
{
	//code for calculating the n-body simulation
	float Fx = 0.0f, Fy = 0.0f;
	for (int j = 0; j < n; j++)
	{
		float dx = p[j]->x - p[i]->x;
		float dy = p[j]->y - p[i]->y;
		float distSqr = dx*dx + dy*dy;
		float invDist = 1.0f / sqrtf(distSqr + SOFTENING);
		float invDist3 = invDist * invDist * invDist;
		Fx += (p[i]->mass) * dx * invDist3;
		Fy += /*(p[i]->mass )*/  dy * invDist3;
	}
	p[i]->vx += dt* Fx;
	p[i]->vy += dt* Fy;
}