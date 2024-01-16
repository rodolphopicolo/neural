#include <stdio.h>

int main();

int main(){
    float samples[24] = {0, 0, 0, 
                        0, 0, 1, 
                        0, 1, 0,
                        1, 0, 0,
                        1, 0, 1,
                        1, 1, 0,
                        0, 1, 1,
                        1, 1, 1};

    float targets[8] = {0, 1, 0, 0, 1, 0, 1, 1};

    FILE *file = fopen("./output/x1_samples.hex", "wb");
    fwrite(samples, 1, sizeof(samples), file);
    fclose(file);

    file = fopen("./output/x1_targets.hex", "wb");
    fwrite(targets, 1, sizeof(targets), file);
    fclose(file);

}