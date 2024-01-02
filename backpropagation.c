#include <stdio.h>
#include <stdlib.h>
#include <math.h>


float derivative_delta_E_delta_a(float *neural, int layers[], int layers_size, float target[], int neuron_index)
{
    int layer_index = layers_size - 1;

    int a_p = a_position(layers, layer_index);
    float a = *(neural + a_p + neuron_index);
    float t = target[neuron_index];
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

void calculate_delta_w(float *neural, int layers[], int layers_size, int layer_index, float target[], float learning_rate, float propagated_value){
    float sum = 0;
    for(int neuron_index = 0; neuron_index < layers[layer_index]; neuron_index++){
        if(layer_index == layers_size - 1){
            float delta_E_delta_a = derivative_delta_E_delta_a(neural, layers, layers_size, target, neuron_index);
            propagated_value = delta_E_delta_a;
        }
        float delta_a_delta_net = derivative_delta_a_delta_net(neural, layers, layer_index, neuron_index);
        
        for(int previous_layer_neuron_index = 0; previous_layer_neuron_index < layers[layer_index - 1]; previous_layer_neuron_index++){
            float delta_net_delta_w = derivative_delta_net_delta_(neural, layers, layer_index, previous_layer_neuron_index);
            float delta_w = propagated_value * delta_a_delta_net * delta_net_delta_w;

            float delta_net_delta_a = derivative_delta_net_delta_a(neural, layers, layer_index, previous_layer_neuron_index, neuron_index);
            sum += propagated_value * delta_a_delta_net * delta_net_delta_a;

        }
    }
    calculate_delta_w(neural, layers, layers_size, layer_index - 1, target, learning_rate, propagated_value);
}






