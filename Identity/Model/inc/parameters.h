#ifndef PARAMETERS_H
#define PARAMETERS_H

#define INPUT_DATA_SIZE (2 << 14)
#define MINI_BATCH_SIZE (2 << 5)
#define EPOCHS (50)
#define LEARNING_RATE (0.1)
#define GAMMA (0.9)
#define BETA1 (0.9)
#define BETA2 (0.999)

#define DIFF_THRESHOLD (0.0001) /* 0.000001 */

#define INPUT_VEC_LOWER_BOUND (-100)
#define INPUT_VEC_UPPER_BOUND (100)

#define PARAMETERS_LOWER_BOUND (0)
#define PARAMETERS_UPPER_BOUND (1)

#endif
