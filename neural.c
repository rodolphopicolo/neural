#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "clock.c"

void load_data(float x[]);
int load_layers(char *arg, int **layers);
float activate(float net);
void call_backpropagation_first_to_last(float *neural, int layers[], int layers_size, float **targets, int target_index, float learning_rate, float momentum);
float backpropagate_first_to_last(float *neural, int layers[], int layers_size, int layer_index, float **target, int target_index, float learning_rate, float momentum, int previous_layer_neuron_index, int layers_exit_check[]);
float derivative_delta_E_delta_a(float *neural, int layers[], int layers_size, float **target, int target_index, int neuron_index);
float derivative_delta_a_delta_net(float *neural, int layers[], int layer_index, int current_layer_neuron_index);
float derivative_delta_net_delta_w(float *neural, int layers[], int layer_index, int previous_layer_neuron_index);
float derivative_delta_net_delta_a(float *neural, int layers[], int layer_index, int previous_layer_neuron_index, int current_layer_neuron_index);
void feed_forward(float *neural, int layers[], int layer_index, int layers_size, float **data, int sample_index, int sample_length);
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
int delta_residual_size(int layers[], int layer_index);
int delta_residual_position(int layers[], int layer_index);
void initialize(float *neural, int layers[], int layers_size);



float derivative_delta_E_delta_a(float *neural, int layers[], int layers_size, float **targets, int target_index, int neuron_index)
{
    int layer_index = layers_size - 1;

    int target_length = layers[layers_size - 1];

    int a_p = a_position(layers, layer_index);
    float a = *(neural + a_p + neuron_index);
    float t = *(*targets + target_index * target_length + neuron_index);
    float delta_E_delta_a = -(t - a);

    return delta_E_delta_a;
}

float derivative_delta_a_delta_net(float *neural, int layers[], int layer_index, int current_layer_neuron_index)
{
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
    int delta_residual = delta_residual_size(layers, layer_index);

    int size = w + delta_w + bias + delta_bias + net + a + delta_residual;
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



int delta_residual_size(int layers[], int layer_index)
{
    return layers[layer_index];
}

int delta_residual_position(int layers[], int layer_index)
{
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
    size_before += a_size(layers, layer_index);

    return size_before;
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

void feed_forward(float *neural, int layers[], int layer_index, int layers_size, float **data, int sample_index, int sample_length)
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
    feed_forward(neural, layers, layer_index + 1, layers_size, data, sample_index, sample_length);
}

struct arguments
{
    char *sample_file_path;
    int sample_length;
    char *target_file_path;
    int target_length;
    int *layers;
    int layers_size;
    float learning_rate;
    float momentum;
};

struct arguments load_arguments(int argument_count, char **arguments)
{
    if (argument_count < 5)
    {
        fprintf(stderr, "\nArguments required: samples file path, sample length, targets file path, target length. Exiting...\n");
        exit(EXIT_FAILURE);
    }
    char *string_part;
    const int BASE = 10;
    struct arguments args;
    for (int i = 1; i < argument_count; i++)
    {
        char *arg = arguments[i];
        if (i == 1)
        {
            args.sample_file_path = arg;
        }
        else if (i == 2)
        {
            args.target_file_path = arg;
        }
        else if (i == 3)
        {
            int *initializer = NULL;
            int **layers = &initializer;
            args.layers_size = load_layers(arg, layers);
            args.layers = *layers;
        } 
        else if (i == 4)
        {
            args.learning_rate = strtof(arg, &string_part);
        }
        else if (i == 5)
        {
            args.momentum = strtof(arg, &string_part);
        }

    }
    args.sample_length = args.layers[0];
    args.target_length = args.layers[args.layers_size - 1];
    return args;
}

int load_layers(char *arg, int **layers)
{
    int layers_size = 0;
    int length = strlen(arg);
    *layers = malloc(sizeof(int) * length);
    char *str_number = malloc(32);
    int number_length = 0;
    for (int i = 0; i < length; i++)
    {
        char c = *(arg + i);
        if (c == ',')
        {
            int neurons = atoi(str_number);
            *(*layers + layers_size) = neurons;
            layers_size++;
            number_length = 0;
        }
        else if (number_length == 0)
        {
            strcpy(str_number, &c);
            number_length++;
        }
        else
        {
            *(str_number + number_length) = c;
            number_length++;
        }
    }
    int neurons = atoi(str_number);
    *(*layers + layers_size) = neurons;
    layers_size++;
    *layers = realloc(*layers, layers_size * sizeof(int));

    return layers_size;
}

int load_input(float **data, char *file_path, int sample_length_float)
{
    int BUFFER_INCREMENT_SIZE_FLOAT = sample_length_float;
    int BUFFER_INCREMENT_SIZE_BYTES = BUFFER_INCREMENT_SIZE_FLOAT * sizeof(float);
    float *buffer = malloc(BUFFER_INCREMENT_SIZE_BYTES);
    int buffer_size_float = BUFFER_INCREMENT_SIZE_FLOAT;
    int buffer_next_free_position_float = 0;
    float sample[sample_length_float];
    FILE *input_file = fopen(file_path, "rb");
    int floats_read;
    int count = 0;

    while ((floats_read = fread(sample, sizeof(float), sample_length_float, input_file)) > 0)
    {
        if (floats_read != sample_length_float)
        {
            printf("Read bytes is diferent of sample length");
            exit(EXIT_FAILURE);
        }

        while (buffer_size_float - buffer_next_free_position_float < floats_read)
        {
            buffer_size_float += BUFFER_INCREMENT_SIZE_FLOAT;
            buffer = realloc(buffer, buffer_size_float * sizeof(float));
        }
        for (int i = 0; i < floats_read; i++)
        {
            *(buffer + buffer_next_free_position_float + i) = sample[i];
        }
        buffer_next_free_position_float += floats_read;
        count++;
    }
    if (buffer_size_float > buffer_next_free_position_float)
    {
        buffer_size_float = buffer_next_free_position_float;
        buffer = realloc(buffer, buffer_size_float * sizeof(float));
    }
    *data = malloc(buffer_size_float * sizeof(float));
    *data = buffer;

    return buffer_next_free_position_float;
}

void show_results(float *neural, float **data, float **targets, int layers[], int layers_size, int sample_index, int sample_length, int target_length, int counter, int epoch, long ch0, long ch1)
{
    printf("\nCounter %4d. Epoch %4d. Sample [", counter, epoch);

    for(int i = 0; i < sample_length && i < 16; i++){
        float value = *(*data + sample_index + i);
        if(i > 0){
            printf(", ");
        }
        printf("%.0f", value);
    }
    float match_value;
    printf("]======================================\nTarget     [");
    int activated_neuron = -1;
    int target_index = sample_index / sample_length;
    for(int neuron = 0; neuron < layers[layers_size - 1] && neuron < 16; neuron++){

        // float t = *(*targets + sample_index * target_length + neuron_index);

        float t = *(*targets + target_index * target_length + neuron);
        if(neuron > 0){
            printf(", ");
        }
        if(t == 1){
            match_value = t;
            activated_neuron = neuron;
            printf("\x1b[32;40m%.5f\x1b[0m", t);

        } else {
            printf("%.5f", t);
        }
        
    }
    const int a_p = a_position(layers, layers_size - 1);

    printf("] \x1b[32;40m%.5f\x1b[0m  \nActivation [", match_value);

    for(int neuron = 0; neuron < layers[layers_size - 1] && neuron < 16; neuron++){
        float a = *(neural + a_p + neuron);
        if(neuron > 0){
            printf(", ");
        }
        if(neuron == activated_neuron){
            match_value = a;
            printf("\x1b[32;40m%.5f\x1b[0m", a);

        } else {
            printf("%.5f", a);
        }
    }

    printf("] \x1b[32;40m%.5f\x1b[0m  \nDiff       [", match_value);


    for(int neuron = 0; neuron < layers[layers_size - 1] && neuron < 16; neuron++){
        float a = *(neural + a_p + neuron);
        float t = *(*targets + target_index * target_length + neuron);
        if(neuron > 0){
            printf(", ");
        }
        float diff = t - a;
        if(diff < 0){
            diff = -diff;
        }
        if(neuron == activated_neuron){
            match_value = diff;
            printf("\x1b[32;40m%.5f\x1b[0m", diff);
        } else {
            printf("%.5f", diff);
        }
    }

    printf("] \x1b[32;40m%.5f\x1b[0m  \nError      [", match_value);

    float sum_error = 0;
    for(int neuron = 0; neuron < layers[layers_size - 1] && neuron < 16; neuron++){
        float a = *(neural + a_p + neuron);
        float t = *(*targets + target_index * target_length + neuron);
        if(neuron > 0){
            printf(", ");
        }
        float error = pow((t-a),2)/2;
        sum_error += error;
        if(neuron == activated_neuron){
            match_value = error;
            printf("\x1b[32;40m%.5f\x1b[0m", error);

        } else {
            printf("%.5f", error);
        }
    }

    printf("] \x1b[32;40m%.5f\x1b[0m              Total error \x1b[32;40m%.5f\x1b[0m", match_value, sum_error);


    // long ch2 = checkpoint();

    // int a_p = a_position(layers, layers_size - 1);
    // float error = 0;
    // float a;
    // float t;
    // for (int neuron = 0; neuron < layers[layers_size - 1]; neuron++)
    // {
    //     a = *(neural + a_p + neuron);
    //     t = *(*targets + (sample_index / sample_length) + neuron);
    //     float e = pow((t - a), 2) / 2;
    //     error += e;
    // }
    // float d0 = *(*data + sample_index + 0);
    // float d1 = *(*data + sample_index + 1);
    // float d2 = *(*data + sample_index + 2);

    // long diff_from_0 = ch2 - ch0;
    // long diff_from_1 = ch2 - ch1;

    // printf("\nCounter %d. Epoch %d. Sample %f,%f,%f. Target: %f. Calculated %f. Error %.8f. Total elapsed %lu. Last elapsed %lu", counter, epoch, d0, d1, d2, t, a, error, diff_from_0, diff_from_1);
}

float backpropagate_first_to_last(float *neural, int layers[], int layers_size, int layer_index, float **target, int target_index, float learning_rate, float momentum, int previous_layer_neuron_index, int layers_exit_check[])
{
    int is_output_layer = (layer_index == layers_size - 1);
    int neurons = layers[layer_index];
    float sum_delta_layer = 0;
    for (int neuron_index = 0; neuron_index < neurons; neuron_index++)
    {

        float delta_layer;
        if (is_output_layer == 1)
        {
            float delta_E_delta_a = derivative_delta_E_delta_a(neural, layers, layers_size, target, target_index, neuron_index);
            delta_layer = delta_E_delta_a;
        }
        else
        {
            float sum_next_delta_layer = backpropagate_first_to_last(neural, layers, layers_size, layer_index + 1, target, target_index, learning_rate, momentum, neuron_index, layers_exit_check);
            delta_layer = sum_next_delta_layer;
        }
        float delta_a_delta_net = derivative_delta_a_delta_net(neural, layers, layer_index, neuron_index);
        delta_layer = delta_layer * delta_a_delta_net;

        if (layers_exit_check[layer_index - 1] == 0)
        {
            int delta_w_p = delta_w_position(layers, layer_index, previous_layer_neuron_index, neuron_index);
            float previous_delta_w = *(neural + delta_w_p);
            float delta_net_delta_w = derivative_delta_net_delta_w(neural, layers, layer_index, previous_layer_neuron_index);
            float delta_w = -learning_rate * delta_layer * delta_net_delta_w + momentum * previous_delta_w;
            int w_p = w_position(layers, layer_index, previous_layer_neuron_index, neuron_index);
            float w = *(neural + w_p);
            w = w + delta_w;
            // printf("\nw:%f, delta_w:%f\n", w, delta_w);
            *(neural + w_p) = w;
            *(neural + delta_w_p) = delta_w;
        }

        if (layers_exit_check[layer_index] == 0)
        {
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

void call_backpropagation_first_to_last(float *neural, int layers[], int layers_size, float **targets, int target_index, float learning_rate, float momentum)
{
    int layers_exit_check[layers_size];
    for (int layer = 0; layer < layers_size; layer++)
    {
        layers_exit_check[layer] = 0;
    }

    int const LAYER_INDEX = 1;
    for (int neuron_index_layer_zero = 0; neuron_index_layer_zero < layers[0]; neuron_index_layer_zero++)
    {
        backpropagate_first_to_last(neural, layers, layers_size, LAYER_INDEX, targets, target_index, learning_rate, momentum, neuron_index_layer_zero, layers_exit_check);
    }
}

void save(float *neural, int neural_size)
{
    char *timestamp = current_time_to_string();
    char *output_file_path = malloc(128);
    sprintf(output_file_path, "%s-%s.bin", "./output/snapshot/neural-snapshot", timestamp);
    FILE *f = fopen(output_file_path, "wb");
    fwrite(neural, neural_size*sizeof(float), 1, f);
    fclose(f);
}

void restore()
{
    //TODO
}

void initialize_with_debugable_values_xor_and_2_3_2(float *neural, int layers[])
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
    *(neural + pos++) = 0.44;
    *(neural + pos++) = 0.41;
    *(neural + pos++) = -0.88;

    pos = bias_position(layers, 2);
    *(neural + pos++) = 0.24;
    *(neural + pos++) = 0.24;
}

void initialize_with_debugable_values_xor_2_3_1(float *neural, int layers[])
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


void calculate_delta_last_layer(float *neural, int layers[], int layers_size, float **targets, int target_index){
    int layer_index = layers_size - 1;
    int neurons = layers[layer_index];

    printf("\nLast layer: ===============================\n");
    int target_length = layers[layers_size - 1];

    for(int neuron_index = 0; neuron_index < neurons; neuron_index++){

            float t = *(*targets + target_index * target_length + neuron_index);
            if(neuron_index > 0){
                printf(", ");
            }
            printf("%.0f", t);


            float delta_E_delta_a = derivative_delta_E_delta_a(neural, layers, layers_size, targets, target_index, neuron_index);
            float delta_a_delta_net = derivative_delta_a_delta_net(neural, layers, layer_index, neuron_index);
            float delta = delta_E_delta_a * delta_a_delta_net;

            int delta_p = delta_residual_position(layers, layer_index) + neuron_index;
            *(neural + delta_p) = delta;
    }
}

void calculate_delta_previous_last_layer(float *neural, int layers[], int layer_index){
    int neurons = layers[layer_index];
    int neurons_next_layer = layers[layer_index + 1];
    for(int neuron_index = 0; neuron_index < neurons; neuron_index++){
        float sum_delta_next_w_current = 0;
        for(int next_layer_neuron_index = 0; next_layer_neuron_index < neurons_next_layer; next_layer_neuron_index++){
            int w_p = w_position(layers, layer_index+1, neuron_index, next_layer_neuron_index);
            float w = *(neural + w_p);

            int delta_next_layer_p = delta_residual_position(layers, layer_index + 1) + next_layer_neuron_index;
            float delta_next_layer = *(neural + delta_next_layer_p);

            sum_delta_next_w_current += delta_next_layer * w;
        }

        float delta_a_delta_net = derivative_delta_a_delta_net(neural, layers, layer_index, neuron_index);
        sum_delta_next_w_current = sum_delta_next_w_current * delta_a_delta_net;
        int delta_p = delta_residual_position(layers, layer_index) + neuron_index;
        *(neural + delta_p) = sum_delta_next_w_current;
    }
}



void backpropagate_last_to_first_2(float *neural, int layers[], int layers_size, int layer_index, float **targets, int target_index, float learning_rate, float momentum){
    if(layer_index == 0){
        return;
    }

    if(layer_index == layers_size - 1){
        calculate_delta_last_layer(neural, layers, layers_size, targets, target_index);
        for(int i = layers_size - 2; i >= 0; i--){
            calculate_delta_previous_last_layer(neural, layers, i);
        }
    }


    int neurons = layers[layer_index];
    int previous_layer_neurons = layers[layer_index - 1];



    float sum_delta_layer = 0;
    for (int neuron_index = 0; neuron_index < neurons; neuron_index++)
    {

        int delta_residual_p = delta_residual_position(layers, layer_index) + neuron_index;
        float delta_layer = *(neural + delta_residual_p);


        int delta_bias_p = delta_bias_position(layers, layer_index);
        float previous_delta_bias = *(neural + delta_bias_p + neuron_index);
        float delta_bias = -learning_rate * delta_layer * 1 + momentum * previous_delta_bias;
        int bias_p = bias_position(layers, layer_index);
        float bias = *(neural + bias_p + neuron_index);
        bias = bias + delta_bias;
        *(neural + bias_p + neuron_index) = bias;
        *(neural + delta_bias_p + neuron_index) = delta_bias;


        for(int previous_layer_neuron_index = 0; previous_layer_neuron_index < previous_layer_neurons; previous_layer_neuron_index++)
        {

            int delta_w_p = delta_w_position(layers, layer_index, previous_layer_neuron_index, neuron_index);
            float previous_delta_w = *(neural + delta_w_p);
            float delta_net_delta_w = derivative_delta_net_delta_w(neural, layers, layer_index, previous_layer_neuron_index);
            float delta_w = -learning_rate * delta_layer * delta_net_delta_w + momentum * previous_delta_w;
            int w_p = w_position(layers, layer_index, previous_layer_neuron_index, neuron_index);
            float w = *(neural + w_p);
            w = w + delta_w;
            *(neural + w_p) = w;
            *(neural + delta_w_p) = delta_w;

        }
    }
    backpropagate_last_to_first_2(neural, layers, layers_size, layer_index - 1, targets, target_index, learning_rate, momentum);
}

void backpropagate_last_to_first(float *neural, int layers[], int layers_size, int layer_index, float **target, int target_index, float learning_rate, float momentum)
{

    if(layer_index == 0){
        return;
    }

    int neurons = layers[layer_index];
    int previous_layer_neurons = layers[layer_index - 1];

    // float residual;
    // if (layer_index < layers_size - 1){
    //     residual = 0;
    //     int delta_residual_p = delta_residual_position(layers, layer_index);
    //     for(int neuron_index = 0; neuron_index < layers[layer_index]; neuron_index++)
    //     {
    //         float delta_residual_neuron = *(neural + delta_residual_p + neuron_index);
    //         residual += delta_residual_neuron;
    //     }
    // }


    float sum_delta_layer = 0;
    for (int neuron_index = 0; neuron_index < neurons; neuron_index++)
    {

        float delta_layer;
        if (layer_index == layers_size - 1)
        {
            float delta_E_delta_a = derivative_delta_E_delta_a(neural, layers, layers_size, target, target_index, neuron_index);
            float delta_a_delta_net = derivative_delta_a_delta_net(neural, layers, layer_index, neuron_index);
            delta_layer = delta_E_delta_a * delta_a_delta_net;

            int delta_residual_p = delta_residual_position(layers, layer_index) + neuron_index;
            *(neural + delta_residual_p) = delta_layer;

        } else {
            int delta_residual_p = delta_residual_position(layers, layer_index);
            delta_layer = *(neural + delta_residual_p + neuron_index);
        }
        


        int delta_bias_p = delta_bias_position(layers, layer_index);
        float previous_delta_bias = *(neural + delta_bias_p + neuron_index);
        float delta_bias = -learning_rate * delta_layer * 1 + momentum * previous_delta_bias;
        int bias_p = bias_position(layers, layer_index);
        float bias = *(neural + bias_p + neuron_index);
        bias = bias + delta_bias;
        *(neural + bias_p + neuron_index) = bias;
        *(neural + delta_bias_p + neuron_index) = delta_bias;


        for(int previous_layer_neuron_index = 0; previous_layer_neuron_index < previous_layer_neurons; previous_layer_neuron_index++)
        {
            float previous_layer_delta_a_delta_net = derivative_delta_a_delta_net(neural, layers, layer_index - 1, previous_layer_neuron_index);
            float delta_net_delta_a = derivative_delta_net_delta_a(neural, layers, layer_index, previous_layer_neuron_index, neuron_index);
            float delta_residual = delta_layer * delta_net_delta_a * previous_layer_delta_a_delta_net;
            int delta_residual_p = delta_residual_position(layers, layer_index - 1) + previous_layer_neuron_index;
            *(neural + delta_residual_p) = delta_residual;



            int delta_w_p = delta_w_position(layers, layer_index, previous_layer_neuron_index, neuron_index);
            float previous_delta_w = *(neural + delta_w_p);
            float delta_net_delta_w = derivative_delta_net_delta_w(neural, layers, layer_index, previous_layer_neuron_index);
            float delta_w = -learning_rate * delta_layer * delta_net_delta_w + momentum * previous_delta_w;
            int w_p = w_position(layers, layer_index, previous_layer_neuron_index, neuron_index);
            float w = *(neural + w_p);
            w = w + delta_w;
            *(neural + w_p) = w;
            *(neural + delta_w_p) = delta_w;


            

        }
    }
    backpropagate_last_to_first(neural, layers, layers_size, layer_index - 1, target, target_index, learning_rate, momentum);
}

int main(int argument_count, char **arguments)
{

    struct arguments args = load_arguments(argument_count, arguments);
    float *data_initializer = NULL;
    float **data = &data_initializer;
    int samples_size = load_input(data, args.sample_file_path, args.sample_length);
    int sample_length = args.sample_length;

    float *targets_initializer = NULL;
    float **targets = &targets_initializer;
    int targets_size = load_input(targets, args.target_file_path, args.target_length);
    int target_length = args.target_length;

    for (int i = 0; i < targets_size; i++)
    {
        float target = *(*targets + i);
    }

    int const layers_size = args.layers_size;
    int layers[layers_size];

    for (int i = 0; i < layers_size; i++)
    {
        layers[i] = *(args.layers + i);
    }

    int neural_size = calculate_size(layers, layers_size);
    float *neural = malloc(neural_size * sizeof(float));

    initialize(neural, layers, layers_size);
    // initialize_with_debugable_values_xor_2_3_1(neural, layers);
    // initialize_with_debugable_values_xor_and_2_3_2(neural, layers);

    int const MAX_EPOCHS = 1000;
    int const FIRST_LAYER_INDEX = 0;
    int const LAST_LAYER_INDEX = layers_size - 1;

    long ch0 = checkpoint();
    int counter = 0;
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        for (int sample_index = 0; sample_index < samples_size; sample_index += sample_length)
        {
            int target_index = sample_index / sample_length;

            feed_forward(neural, layers, FIRST_LAYER_INDEX, layers_size, data, sample_index, sample_length);
            long ch1 = checkpoint();
            int layer_index = layers_size - 1;
            backpropagate_last_to_first_2(neural, layers, layers_size, layer_index, targets, target_index, args.learning_rate, args.momentum);
            // call_backpropagation_first_to_last(neural, layers, layers_size, targets, target_index, LEARNING_RATE, MOMENTUM);

            if (counter % 1 == 0)
            {
                show_results(neural, data, targets, layers, layers_size, sample_index, sample_length, target_length, counter, epoch, ch0, ch1);
            }
            counter++;
        }
        // save(neural, neural_size);
    }
}