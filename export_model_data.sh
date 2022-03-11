#!/bin/bash

unset -v container_id
unset -v platform

while getopts "c:p:" opt; do
    case ${opt} in
        c )
        container_id=$OPTARG
        ;;

        p )
        platform=$OPTARG
        ;;

        \? )
        #print option error
        echo "Invalid option: $OPTARG" 1>&2
        ;;

        : )
        #print argument error
        echo "Invalid option: $OPTARG requires an argument" 1>&2
        ;;
    esac
done

: ${container_id:?Missing -c}
: ${platform:?Missing -p}

if [ ! -d /tmp/model_output_path_${container_id} ]; then
    sudo docker cp $container_id:/opt/triton-model-analyzer/output_model_repository /tmp/model_output_path_${container_id}
    sudo chmod -R a+rwX /tmp/model_output_path_${container_id}
fi

tf_models=()
tflite_models=()

# Get names of all tflite and tf models in test repo
for d in admission_controller/tests/triton_model_repo/*/ ; do
    model_name=$(basename $d)
    if [ -f $d/1/*graphdef ]; then
        tf_models+=($model_name)
    elif [ -f $d/1/*tflite ]; then
        tflite_models+=($model_name)
    fi
done

echo "Tensorflow models: ${tf_models[@]}"
echo "Tflite models: ${tflite_models[@]}"

for model_name in ${tf_models[@]}; do
    docker exec $container_id model-analyzer analyze --analysis-models $model_name --inference-output-fields model_name,batch_size,concurrency,model_config_path,instance_group,dynamic_batch_sizes,satisfies_constraints,perf_throughput,perf_latency_p99,cpu_used_ram,cpu_util,backend_parameter/TF_NUM_INTRA_THREADS,backend_parameter/TF_NUM_INTER_THREADS
    if [ $? -eq 0 ]; then
        docker cp $container_id:/opt/triton-model-analyzer/results/metrics-model-inference.csv ~/code/admission_controller/admission_controller/tests/triton_model_repo/$model_name/metrics-model-inference-$platform.csv
    else
        echo "No profiling data found for tf model: $model_name"
    fi
done

for model_name in ${tflite_models[@]}; do
    docker exec $container_id model-analyzer analyze --analysis-models $model_name --inference-output-fields model_name,batch_size,concurrency,model_config_path,instance_group,dynamic_batch_sizes,satisfies_constraints,perf_throughput,perf_latency_p99,cpu_used_ram,cpu_util,backend_parameter/tflite_num_threads
    if [ $? -eq 0 ]; then
        docker cp $container_id:/opt/triton-model-analyzer/results/metrics-model-inference.csv ~/code/admission_controller/admission_controller/tests/triton_model_repo/$model_name/metrics-model-inference-$platform.csv
        python3 add_tflite_data.py -r /tmp/model_output_path_${container_id} -m ~/code/admission_controller/admission_controller/tests/triton_model_repo/$model_name/metrics-model-inference-$platform.csv -i    
    else
        echo "No profiling data found for tflite model: $model_name"
    fi
done






 
