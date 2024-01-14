#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void load_data(float x[]);
float activate(float net);
float backpropagate(float *neural, int layers[], int layers_size, int layer_index, float **target, int target_index, float learning_rate, float momentum, int previous_layer_neuron_index, int layers_exit_check[]);
float derivative_delta_E_delta_a(float *neural, int layers[], int layers_size, float **target, int target_index, int neuron_index);
float derivative_delta_a_delta_net(float *neural, int layers[], int layer_index, int current_layer_neuron_index);
float derivative_delta_net_delta_w(float *neural, int layers[], int layer_index, int previous_layer_neuron_index);
float derivative_delta_net_delta_a(float *neural, int layers[], int layer_index, int previous_layer_neuron_index, int current_layer_neuron_index);
void propagate(float *neural, int layers[], int layer_index, int layers_size, float **data, int sample_index, int sample_length);
void sample_to_first_layer(float *neural, int layers[], float **data, int sample_index, int sample_length);
void calculate_net(float *neural, int layers[], int layer_index);
void calculate_activation(float *neural, int layers[], int layer_index);
int layer_size(int layers[], int layer_index);
int calculate_size(int layers[], int size);
int w_position(int layers[], int layer_index, int previous_layer_neuron_index, int current_layer_neuron_index);
int w_size(int layers[], int layer_index);
int bias_position(int layers[], int layer_index);
int bias_size(int layers[], int layer_index);
int net_position(int layers[], int layer_index);
int net_size(int layers[], int layer_index);
int a_position(int layers[], int layer_index);
int a_size(int layers[], int layer_index);
int delta_w_position(int layers[], int layer_index, int previous_layer_neuron_index, int current_layer_neuron_index);
int delta_w_size(int layers[], int layer_index);
int delta_bias_position(int layers[], int layer_index);
int delta_bias_size(int layers[], int layer_index);
void initialize(float *neural, int layers[], int layers_size);

float derivative_delta_E_delta_a(float *neural, int layers[], int layers_size, float **target, int target_index, int neuron_index)
{
    int layer_index = layers_size - 1;

    int a_p = a_position(layers, layer_index);
    float a = *(neural + a_p + neuron_index);
    float t = *(*target + target_index + neuron_index);
    float delta_E_delta_a = -(t - a);
    
    return delta_E_delta_a;
}

float derivative_delta_a_delta_net(float *neural, int layers[], int layer_index, int current_layer_neuron_index){
    int a_p = a_position(layers, layer_index) + current_layer_neuron_index;
    float a = *(neural + a_p);
    float delta_a_delta_net = (1 - a) * a;
    return delta_a_delta_net;
}

float derivative_delta_net_delta_w(float *neural, int layers[], int layer_index, int previous_layer_neuron_index)
{
    int a_pos = a_position(layers, layer_index - 1);
    float delta_net_delta_w = *(neural + a_pos + previous_layer_neuron_index);
    return delta_net_delta_w;
}

float derivative_delta_net_delta_a(float *neural, int layers[], int layer_index, int previous_layer_neuron_index, int current_layer_neuron_index)
{
    int w_p = w_position(layers, layer_index, previous_layer_neuron_index, current_layer_neuron_index);
    float delta_net_delta_a = *(neural + w_p);
    return delta_net_delta_a;
}

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

float activate(float net)
{
    float a = 1 / (1 + exp(-net));
    return a;
}

void sample_to_first_layer(float *neural, int layers[], float **data, int sample_index, int sample_length)
{
    int const FIRST_LAYER_INDEX = 0;
    int pos = a_position(layers, FIRST_LAYER_INDEX);
    int size = a_size(layers, FIRST_LAYER_INDEX);
    if (size != sample_length)
    {
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++)
    {
        *(neural + pos + i) = *(*data + sample_index + i);
    }
}

void calculate_net(float *neural, int layers[], int layer_index)
{

    int previous_layer_size = layers[layer_index - 1];

    int a_p = a_position(layers, layer_index - 1);
    int a_s = a_size(layers, layer_index - 1);

    int w_p = w_position(layers, layer_index, 0, 0);

    int bias_p = bias_position(layers, layer_index);

    int net_p = net_position(layers, layer_index);
    int net_s = net_size(layers, layer_index);

    for (int i = 0; i < net_s; i++)
    {
        float net = 0;
        for (int j = 0; j < a_s; j++)
        {
            float a = *(neural + a_p + j);
            float w = *(neural + w_p + (i * previous_layer_size + j));
            net += a * w;
        }
        float bias = *(neural + bias_p + i);
        net += bias;
        *(neural + net_p + i) = net;
    }
}

void calculate_activation(float *neural, int layers[], int layer_index)
{
    int net_p = net_position(layers, layer_index);
    int net_s = net_size(layers, layer_index);

    int a_p = a_position(layers, layer_index);
    int a_s = a_size(layers, layer_index);

    if (net_s != a_s)
    {
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < net_s; i++)
    {
        float net = *(neural + net_p + i);
        float a = activate(net);
        *(neural + a_p + i) = a;
    }
}

int layer_size(int layers[], int layer_index)
{
    int w = w_size(layers, layer_index);
    int delta_w = delta_w_size(layers, layer_index);
    int bias = bias_size(layers, layer_index);
    int delta_bias = delta_bias_size(layers, layer_index);
    int net = net_size(layers, layer_index);
    int a = a_size(layers, layer_index);

    int size = w + delta_w + bias + delta_bias + net + a;
    return size;
}

int calculate_size(int layers[], int size)
{
    int total_size = 0;
    for (int i = 0; i < size; i++)
    {
        total_size += layer_size(layers, i);
    }
    return total_size;
}

int w_position(int layers[], int layer_index, int previous_layer_neuron_index, int current_layer_neuron_index)
{
    if (layer_index == 0)
    {
        return -1;
    }
    int size_before = 0;
    for (int i = 0; i < layer_index; i++)
    {
        size_before += layer_size(layers, i);
    }

    int w_index = current_layer_neuron_index * layers[layer_index - 1] + previous_layer_neuron_index;
    int position = size_before + w_index;

    return position;
}
int w_size(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return 0;
    }
    return layers[layer_index - 1] * layers[layer_index];
}

int delta_w_position(int layers[], int layer_index, int previous_layer_neuron_index, int current_layer_neuron_index)
{
    if (layer_index == 0)
    {
        return -1;
    }
    int size_before = 0;
    for (int i = 0; i < layer_index; i++)
    {
        size_before += layer_size(layers, i);
    }
    size_before += w_size(layers, layer_index);

    int delta_w_index = current_layer_neuron_index * layers[layer_index - 1] + previous_layer_neuron_index;
    int position = size_before + delta_w_index;

    return position;
}
int delta_w_size(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return 0;
    }
    return layers[layer_index - 1] * layers[layer_index];
}

int bias_position(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return -1;
    }
    int size_before = 0;
    for (int i = 0; i < layer_index; i++)
    {
        size_before += layer_size(layers, i);
    }
    size_before += w_size(layers, layer_index);
    size_before += delta_w_size(layers, layer_index);
    return size_before;
}

int bias_size(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return 0;
    }
    return layers[layer_index];
}

int delta_bias_position(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return -1;
    }
    int size_before = 0;
    for (int i = 0; i < layer_index; i++)
    {
        size_before += layer_size(layers, i);
    }
    size_before += w_size(layers, layer_index);
    size_before += delta_w_size(layers, layer_index);
    size_before += bias_size(layers, layer_index);

    return size_before;
}

int delta_bias_size(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return 0;
    }
    return layers[layer_index];
}

int net_position(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return -1;
    }
    int size_before = 0;
    for (int i = 0; i < layer_index; i++)
    {
        size_before += layer_size(layers, i);
    }
    size_before += w_size(layers, layer_index);
    size_before += delta_w_size(layers, layer_index);
    size_before += bias_size(layers, layer_index);
    size_before += delta_bias_size(layers, layer_index);
    return size_before;
}

int net_size(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return 0;
    }
    return layers[layer_index];
}
int a_position(int layers[], int layer_index)
{
    if (layer_index == 0)
    {
        return 0;
    }
    int size_before = 0;
    for (int i = 0; i < layer_index; i++)
    {
        size_before += layer_size(layers, i);
    }
    size_before += w_size(layers, layer_index);
    size_before += delta_w_size(layers, layer_index);
    size_before += bias_size(layers, layer_index);
    size_before += delta_bias_size(layers, layer_index);
    size_before += net_size(layers, layer_index);
    return size_before;
}
int a_size(int layers[], int layer_index)
{
    return layers[layer_index];
}

void initialize(float *neural, int layers[], int layers_size)
{
    srandom(time(NULL));
    float r;
    int position;
    int size;
    int last_position;

    for (int i = 1; i < layers_size; i++)
    {
        position = w_position(layers, i, 0, 0);
        size = w_size(layers, i);
        last_position = position + size - 1;

        for (int j = position; j <= last_position; j++)
        {
            r = random();
            r = r - (RAND_MAX / 2);
            r = r / RAND_MAX;
            *(neural + j) = r;
        }

        position = bias_position(layers, i);
        size = bias_size(layers, i);
        last_position = position + size - 1;

        for (int j = position; j <= last_position; j++)
        {
            r = random();
            r = r - (RAND_MAX / 2);
            r = r / RAND_MAX;
            *(neural + j) = r;
        }
    }
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

void propagate(float *neural, int layers[], int layer_index, int layers_size, float **data, int sample_index, int sample_length)
{
    if (layer_index >= layers_size)
    {
        exit(EXIT_FAILURE);
    }
    if (layer_index == 0)
    {
        sample_to_first_layer(neural, layers, data, sample_index, sample_length);
    }
    else
    {
        calculate_net(neural, layers, layer_index);
        calculate_activation(neural, layers, layer_index);
    }

    if (layer_index + 1 >= layers_size)
    {
        return;
    }
    propagate(neural, layers, layer_index + 1, layers_size, data, sample_index, sample_length);
}

float backpropagate_2_3_1(float *neural, int layers[], int layers_size, int layer_index, float target[], int target_index, float learning_rate, float momentum, int previous_layer_neuron_index){
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
                float t_k = target[target_index + k];
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


float backpropagate(float *neural, int layers[], int layers_size, int layer_index, float **target, int target_index, float learning_rate, float momentum, int previous_layer_neuron_index, int layers_exit_check[]){
    int is_output_layer = (layer_index == layers_size - 1);
    int neurons = layers[layer_index];
    float sum_delta_layer = 0;
    for(int neuron_index = 0; neuron_index < neurons; neuron_index++){
        float delta_a_delta_net = derivative_delta_a_delta_net(neural, layers, layer_index, neuron_index);
        float delta_layer;
        if(is_output_layer == 1){
            float delta_E_delta_a = derivative_delta_E_delta_a(neural, layers, layers_size, target, target_index, neuron_index);    
            delta_layer = delta_E_delta_a * delta_a_delta_net;
        } else {
            float sum_next_delta_layer = backpropagate(neural, layers, layers_size, layer_index+1, target, target_index, learning_rate, momentum, neuron_index, layers_exit_check);
            delta_layer = sum_next_delta_layer * delta_a_delta_net;
        }
        float delta_net_delta_w = derivative_delta_net_delta_w(neural, layers, layer_index, previous_layer_neuron_index);

        int delta_w_p = delta_w_position(layers, layer_index, previous_layer_neuron_index, neuron_index);
        float previous_delta_w = *(neural + delta_w_p);

        float delta_w = -learning_rate * delta_layer * delta_net_delta_w + momentum * previous_delta_w;
        int w_p = w_position(layers, layer_index, previous_layer_neuron_index, neuron_index);
        float w = *(neural + w_p);
        w = w + delta_w;

        if(layers_exit_check[layer_index-1] == 0){
            // printf("\nw:%f, delta_w:%f\n", w, delta_w);
            *(neural + w_p) = w;
            *(neural + delta_w_p) = delta_w;
        }


        if(layers_exit_check[layer_index] == 0){
            int delta_bias_p = delta_bias_position(layers, layer_index);
            float previous_delta_bias = *(neural + delta_bias_p + neuron_index);
            float delta_bias = -learning_rate * delta_layer * 1 + momentum * previous_delta_bias;
            int bias_p = bias_position(layers, layer_index);
            float bias = *(neural + bias_p + neuron_index);
            bias = bias + delta_bias;
            // printf("\nbias:%f, delta_bias:%f\n", bias, delta_bias);
            *(neural + bias_p + neuron_index) = bias;
            *(neural + delta_bias_p + neuron_index) = delta_bias;
        }

        float delta_net_delta_a = derivative_delta_net_delta_a(neural, layers, layer_index, previous_layer_neuron_index, neuron_index);
        sum_delta_layer += delta_layer * delta_net_delta_a;
    }

    layers_exit_check[layer_index] = 1;
    return sum_delta_layer;
}


struct arguments {
    char *sample_file_path;
    int sample_length;
    char *target_file_path;
    int target_length;
};

struct arguments load_arguments(int argument_count, char **arguments){
    printf("\n\n");
    if(argument_count < 5){
        fprintf(stderr, "\nArguments required: samples file path, sample length, targets file path, target length. Exiting...\n");
        exit(EXIT_FAILURE);
    }
    char *string_part;
    const int BASE = 10;
    struct arguments args;
    for(int i = 1; i < argument_count; i++){
        char *arg = arguments[i];
        printf("\nArgument %d: [%s].", i, arg);

        if(i == 1){
            args.sample_file_path = arg;
        } else if(i == 2){
            args.sample_length = strtol(arg, &string_part, BASE);
        } else if(i == 3){
            args.target_file_path = arg;
        } else if(i == 4){
            args.target_length = strtol(arg, &string_part, BASE);
        }

    }
    printf("\n\n");
    return args;
}

int load_input(float **data, char *file_path, int length){
    const int BUFFER_INCREMENT_SIZE = 3;
    float *buffer = malloc(BUFFER_INCREMENT_SIZE);
    int buffer_size = BUFFER_INCREMENT_SIZE;
    int buffer_next_free_position = 0;
    float sample[length];
    FILE *input_file = fopen(file_path, "rb");
    int bytes_read;
    int count = 0;
    float *d = *data;
    while((bytes_read = fread(sample, sizeof(float), length, input_file))==length){

        if(buffer_size - buffer_next_free_position < bytes_read){
             buffer_size += BUFFER_INCREMENT_SIZE;
            buffer = realloc(buffer, buffer_size);
        }

        printf("\nSample %d: ", count);
        for(int i = 0; i < length; i++){
            if(i > 0){
                printf(", ");
            }
            printf("%.4f", sample[i]);
            *(buffer + length * count + i) = sample[i];
        }
        buffer_next_free_position += bytes_read;
        count++;
        
    }
    if(buffer_size > buffer_next_free_position){
        buffer_size = buffer_next_free_position;
        buffer = realloc(buffer, buffer_size);
    }
    *data = buffer;
    return count*length;
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

    initialize(neural, layers, layers_size);
    initialize_with_debugable_values(neural, layers);

    // int samples_size = 8;
    // int sample_length = 2;
    // float data[samples_size];
    // load_data(data);

    // int targets_size = 4;
    // int target_length = 1;
    // float targets[targets_size];
    // load_target(targets);

    int const MAX_EPOCHS = 500;
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

            // int neuron_index_layer_zero = 0;
            // backpropagate_2_3_1(neural, layers, layers_size, 0, target, target_index, LEARNING_RATE, MOMENTUM, neuron_index_layer_zero);
            int const LAYER_INDEX = 1;
            for(int neuron_index_layer_zero = 0; neuron_index_layer_zero < layers[0]; neuron_index_layer_zero++){
                backpropagate(neural, layers, layers_size, LAYER_INDEX, targets, target_index, LEARNING_RATE, MOMENTUM, neuron_index_layer_zero, layers_exit_check);    
            }

            if(epoch == MAX_EPOCHS - 1 || epoch > -1){
                int a_p = a_position(layers, layers_size - 1);
                float a = *(neural + a_p);
                float t = *(*targets + (sample_index/sample_length));
                float d0 = *(*data + (sample_index/sample_length));
                float d1 = *(*data + (sample_index/sample_length+1));
                float error = 1.0/2*(t - a)*(t - a);
                printf("\nCounter %d. Epoch %d. Sample %f,%f. Target: %f. Calculated %f. Error %.8f.\n", counter, epoch, d0, d1, t, a, error);
            }
            counter++;

        }
    }
}