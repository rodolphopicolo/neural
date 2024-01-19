#include <stdio.h>
#include "neural.c"

int main(int argument_count, char **arguments);



void load_target(float t[])
{
    t[0] = 0;
    t[1] = 1;
    t[2] = 1;
    t[3] = 0;
}

void load_data(float x[])
{
    x[0] = 0;
    x[1] = 0;
    x[2] = 0;
    x[3] = 1;
    x[4] = 1;
    x[5] = 0;
    x[6] = 1;
    x[7] = 1;
}


void initialize_with_debugable_values(float *neural, int layers[])
{
    int pos = w_position(layers, 1, 0, 0);
    *(neural + pos++) = 0.19;
    *(neural + pos++) = 0.74;
    *(neural + pos++) = -0.43;
    *(neural + pos++) = -0.11;
    *(neural + pos++) = 0.33;
    *(neural + pos++) = 0.14;

    pos = bias_position(layers, 1);
    *(neural + pos++) = 0.36;
    *(neural + pos++) = 0.82;
    *(neural + pos++) = -0.31;

    pos = w_position(layers, 2, 0, 0);
    *(neural + pos++) = 0.44;
    *(neural + pos++) = 0.41;
    *(neural + pos++) = -0.88;

    pos = bias_position(layers, 2);
    *(neural + pos++) = 0.24;
}

float backpropagate_2_3_1(float *neural, int layers[], int layers_size, int layer_index, float **target, int target_index, float learning_rate, float momentum, int previous_layer_neuron_index){
    int const k_size = layers[2];
    int const j_size = layers[1];
    int calculate_bias_k[k_size];
    int calculate_bias_j[j_size];
    for(int k = 0; k < layers[2]; k++){
        calculate_bias_k[k] = 1;
    }
    for(int j = 0; j < layers[1]; j++){
        calculate_bias_j[j] = 1;
    }

    for(int i = 0; i < layers[0]; i++){
        float a_i = *(neural + a_position(layers, 0) + i);
        float sum_delta_j = 0;
        for(int j = 0; j < layers[1]; j++){
            float a_j = *(neural + a_position(layers, 1) + j);
            float sum_delta_k = 0;
            for(int k = 0; k < layers[2]; k++){
                float a_k = *(neural + a_position(layers, 2) + k);
                float t_k = *(*target +target_index + k);
                float delta_k = (t_k - a_k)*a_k*(1-a_k);
                float previous_delta_w_k_j = *(neural + delta_w_position(layers, 2, j, k));
                float delta_w_k_j = learning_rate * delta_k * a_j + momentum * previous_delta_w_k_j;
                float w_k_j = *(neural + w_position(layers, 2, j, k));
                sum_delta_k += delta_k * w_k_j;
                w_k_j += delta_w_k_j;

                if(i == 0){
                    *(neural + w_position(layers, 2, j, k)) = w_k_j;
                    *(neural + delta_w_position(layers, 2, j, k)) = delta_w_k_j;
                }

                /*
                    Bias is not dependent on j, so, the block below should be
                    executed just once. Executing every interaction the previous_delta_bias_k
                    interfere to the current delta bias, but maybe this is making the conversion
                    be faster.
                    It should be executed once per k within the whole backpropagation routine.
                */
               if(calculate_bias_k[k] == 1){
                    float previous_delta_bias_k = *(neural + delta_bias_position(layers, 2) + k);
                    float delta_bias_k = learning_rate * delta_k + momentum * previous_delta_bias_k;
                    float bias_k = *(neural + bias_position(layers, 2) + k);
                    bias_k += delta_bias_k;
                    *(neural + bias_position(layers, 2) + k) = bias_k;
                    *(neural + delta_bias_position(layers, 2) + k) = delta_bias_k;

                    calculate_bias_k[k] = 0;
               }
                
                
            }
            float delta_j = sum_delta_k * a_j * (1 - a_j);
            float previous_delta_w_j_i = *(neural + delta_w_position(layers, 1, i, j));
            float delta_w_j_i = learning_rate * delta_j * a_i + momentum * previous_delta_w_j_i;
            float w_j_i = *(neural + w_position(layers, 1, i, j));
            sum_delta_j += delta_j * w_j_i;
            w_j_i += delta_w_j_i;
            *(neural + w_position(layers, 1, i, j)) = w_j_i;
            *(neural + delta_w_position(layers, 1, i, j)) = delta_w_j_i;


            /*
                The same thing happens here, related to the bias_k, described above.
                It should be executed once per j within the whole backpropagation routine.
            */
            if(calculate_bias_j[j] == 1){
                float previous_delta_bias_j = *(neural + delta_bias_position(layers, 1) + j);
                float delta_bias_j = learning_rate * delta_j + momentum * previous_delta_bias_j;
                float bias_j = *(neural + bias_position(layers, 1) + j);
                bias_j += delta_bias_j;
                *(neural + bias_position(layers, 1) + j) = bias_j;
                *(neural + delta_bias_position(layers, 1) + j) = delta_bias_j;

                calculate_bias_j[j] = 0;
            }
            
        }
    }
    return 0;

}

int main(int argument_count, char **arguments){

    struct arguments args = load_arguments(argument_count, arguments);
    float *data_initializer = NULL;
    float **data = &data_initializer;
    int samples_size = load_input(data, args.sample_file_path, args.sample_length);
    int sample_length = args.sample_length;

    float *targets_initializer = NULL;
    float **targets = &targets_initializer;
    int targets_size = load_input(targets, args.target_file_path, args.target_length);
    int target_length = args.target_length;

    int layers[3] = {2, 3, 1};
    int layers_size = sizeof(layers) / sizeof(layers[0]);


    float LEARNING_RATE = 0.45;
    float MOMENTUM = 0.9;

    int neural_size = calculate_size(layers, layers_size);
    float *neural = malloc(neural_size * sizeof(float));

    initialize_with_debugable_values(neural, layers);

    int const MAX_EPOCHS = 1000;
    int const FIRST_LAYER_INDEX = 0;
    int const LAST_LAYER_INDEX = layers_size - 1;

    int counter = 0;
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        for (int sample_index = 0; sample_index < samples_size; sample_index += sample_length)
        {
            int target_index = sample_index/sample_length;

            propagate(neural, layers, FIRST_LAYER_INDEX, layers_size, data, sample_index, sample_length);

            int layers_exit_check[layers_size];
            for(int layer = 0; layer < layers_size; layer++){
                layers_exit_check[layer] = 0;
            }

            int neuron_index_layer_zero = 0;
            backpropagate_2_3_1(neural, layers, layers_size, 0, targets, target_index, LEARNING_RATE, MOMENTUM, neuron_index_layer_zero);

            if(epoch == MAX_EPOCHS - 1 || epoch > -1){
                int a_p = a_position(layers, layers_size - 1);
                float a = *(neural + a_p);
                float t = *(*targets + (sample_index/sample_length));
                float d0 = *(*data + (sample_index/sample_length));
                float d1 = *(*data + (sample_index/sample_length+1));
                float error = 1.0/2*(t - a)*(t - a);
                printf("\nCounter %d. Epoch %d. Sample %f,%f. Target: %f. Calculated %f. Error %.8f.", counter, epoch, d0, d1, t, a, error);
            }
            counter++;

        }
    }
}