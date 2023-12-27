#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void load_data(float x[]);
float activate(float net);
void back_propagate();
void propagate(float *neural, int layers[], int layer_index, int layers_size, float data[], int sample_first_index, int sample_size);
void sample_to_first_layer_output(float *neural, int layers[], float data[], int sample_first_index, int sample_size);
void calculate_net(float *neural, int layers[], int layer_index);
void calculate_activation(float *neural, int layers[], int layer_index);
int layer_size(int layers[], int layer_index);
int calculate_size(int layers[], int size);
int w_position(int layers[], int layer_index);
int w_size(int layers[], int layer_index);
int bias_position(int layers[], int layer_index);
int bias_size(int layers[], int layer_index);
int net_position(int layers[], int layer_index);
int net_size(int layers[], int layer_index);
int a_position(int layers[], int layer_index);
int a_size(int layers[], int layer_index);
void initialize(float *neural, int layers[], int layers_size);


void load_data(float x[]){
    x[0] = 0;
    x[1] = 0;
    x[2] = 0;
    x[3] = 1;
    x[4] = 1;
    x[5] = 0;
    x[6] = 1;
    x[7] = 1;
}

float activate(float net){
    float a = 1/(1+exp(-net));
    return a;
}

void back_propagate(){

}

void sample_to_first_layer_output(float *neural, int layers[], float data[], int sample_first_index, int sample_size){
    int const FIRST_LAYER_INDEX = 0;
    int pos = a_position(layers, FIRST_LAYER_INDEX);
    int size = a_size(layers, FIRST_LAYER_INDEX);
    if(size != sample_size){
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < size; i++){
        *(neural + pos + i) = data[sample_first_index + i];
    }
}

void calculate_net(float *neural, int layers[], int layer_index){

    int previous_layer_size = layers[layer_index - 1];

    int a_p = a_position(layers, layer_index - 1);
    int a_s = a_size(layers, layer_index - 1);

    int w_p = w_position(layers, layer_index);
    int w_s = w_size(layers, layer_index);

    int bias_p = bias_position(layers, layer_index);

    int net_p = net_position(layers, layer_index);
    int net_s = net_size(layers, layer_index);

    for(int i = 0; i < net_s; i++){
        float net = 0;
        for(int j = 0; j < a_s; j++){
            float a = *(neural + a_p + j);
            float w = *(neural + w_p + (i * previous_layer_size + j));
            net += a * w;
        }
        float bias = *(neural + bias_p + i);
        net += bias;
        *(neural + net_p + i) = net;
    }
}

void calculate_activation(float *neural, int layers[], int layer_index){
    int net_p = net_position(layers, layer_index);
    int net_s = net_size(layers, layer_index);

    int a_p = a_position(layers, layer_index);
    int a_s = a_size(layers, layer_index);

    if(net_s != a_s){
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < net_s; i++){
        float net = *(neural + net_p + i);
        float a = activate(net);
        *(neural + a_p + i) = a;
    }
}

int layer_size(int layers[], int layer_index){
    int w = w_size(layers, layer_index);
    int bias = bias_size(layers, layer_index);
    int net = net_size(layers, layer_index);
    int a = a_size(layers, layer_index);
    int size = w + bias + net + a;
    return size;
}

int calculate_size(int layers[], int size){
    int total_size = 0;
    for(int i = 0; i < size; i++){
        total_size += layer_size(layers, i);
    }
    return total_size;
}



int w_position(int layers[], int layer_index){
    if(layer_index == 0){
        return -1;
    }
    int size_before = 0;
    for(int i = 0; i < layer_index; i++){
        size_before += layer_size(layers, i);
    }
    return size_before;
}
int w_size(int layers[], int layer_index){
    if(layer_index == 0){
        return 0;
    }
    return layers[layer_index-1]*layers[layer_index];
}

int bias_position(int layers[], int layer_index){
    if(layer_index == 0){
        return -1;
    }
    int size_before = 0;
    for(int i = 0; i < layer_index; i++){
        size_before += layer_size(layers, i);
    }
    size_before += w_size(layers, layer_index);
    return size_before;
}

int bias_size(int layers[], int layer_index){
    if(layer_index == 0){
        return 0;
    }
    return layers[layer_index];
}
int net_position(int layers[], int layer_index){
    if(layer_index == 0){
        return -1;
    }
    int size_before = 0;
    for(int i = 0; i < layer_index; i++){
        size_before += layer_size(layers, i);
    }
    size_before += w_size(layers, layer_index);
    size_before += bias_size(layers, layer_index);
    return size_before;
}

int net_size(int layers[], int layer_index){
    if(layer_index == 0){
        return 0;
    }
    return layers[layer_index];
}
int a_position(int layers[], int layer_index){
    if(layer_index == 0){
        return 0;
    }
    int size_before = 0;
    for(int i = 0; i < layer_index; i++){
        size_before += layer_size(layers, i);
    }
    size_before += w_size(layers, layer_index);
    size_before += bias_size(layers, layer_index);
    size_before += net_size(layers, layer_index);
    return size_before;
}
int a_size(int layers[], int layer_index){
    return layers[layer_index];
}

void initialize(float *neural, int layers[], int layers_size){
    srandom(time(NULL));
    float r;
    int position;
    int size;
    int last_position;

    for(int i = 1; i < layers_size; i++){
        position = w_position(layers, i);
        size = w_size(layers, i);
        last_position = position + size - 1;

        for(int j = position; j <= last_position; j++){
            r = random();
            r =  r - (RAND_MAX / 2);
            r = r / RAND_MAX;
            *(neural + j) = r;
        }

        position = bias_position(layers, i);
        size = bias_size(layers, i);
        last_position = position + size - 1;

        for(int j = position; j <= last_position; j++){
            r = random();
            r =  r - (RAND_MAX / 2);
            r = r / RAND_MAX;
            *(neural + j) = r;
        }
    }
}

void initialize_with_debugable_values(float *neural, int layers[]){
    int pos = w_position(layers, 1);
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

    pos = w_position(layers, 2);
    *(neural + pos++) = 0.44;
    *(neural + pos++) = 0.41;
    *(neural + pos++) = -0.88;

    pos = bias_position(layers, 2);
    *(neural + pos++) = 0.24;
}

void propagate(float *neural, int layers[], int layer_index, int layers_size, float data[], int sample_first_index, int sample_size){
    if(layer_index >= layers_size){
        exit(EXIT_FAILURE);
    }
    if(layer_index == 0){
        sample_to_first_layer_output(neural, layers, data, sample_first_index, sample_size);
    } else {
        calculate_net(neural, layers, layer_index);
        calculate_activation(neural, layers, layer_index);
    }

    if(layer_index + 1 >= layers_size){
        return;
    }
    propagate(neural, layers, layer_index+1, layers_size, data, sample_first_index, sample_size);
}



int main(int argument_count, char **arguments){
    int layers[3] = {2, 3, 1};
    int layers_size = sizeof(layers)/sizeof(layers[0]);

    float alpha = 0.45;
    float momentum = 0.9;


    int neural_size = calculate_size(layers, layers_size);
    float *neural = malloc(neural_size*sizeof(float));

    initialize(neural, layers, layers_size);
    initialize_with_debugable_values(neural, layers);


    int total_size = 8;
    int sample_size = 2;
    float data[total_size];
    load_data(data);

    int const MAX_EPOCHS = 10;
    int const FIRST_LAYER_INDEX = 0;
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++){

        for(int sample_first_index = 0; sample_first_index < total_size; sample_first_index+=sample_size){
            propagate(neural, layers, FIRST_LAYER_INDEX, layers_size, data, sample_first_index, sample_size);
            printf("\n");
        }
    }
}