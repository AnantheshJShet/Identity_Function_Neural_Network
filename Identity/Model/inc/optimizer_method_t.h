#ifndef OPTIMIZER_METHOD_T_H
#define OPTIMIZER_METHOD_T_H

typedef enum Optimizer_Method_Tag
{
	OPTIMIZER_GD,
	OPTIMIZER_GD_WITH_MOMENTUM,
	OPTIMIZER_RMS,
	OPTIMIZER_ADAM
}Optimizer_Method_T;

#endif
