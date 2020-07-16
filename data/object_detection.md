## **Training : on docker** 



installing for start

1. install docker

   curl -sSL https://get.docker.com/ | sh 

2. install nvidia docker 

   **If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers****

   1. docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
   2. sudo apt-get purge -y nvidia-docker

   **Add the package repositories**

   3. curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \

      sudo apt-key add - 

   4. distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

   5. curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   6. sudo apt-get update

   #### Install nvidia-docker2 and reload the Docker daemon configuration

   6. sudo apt-get install -y nvidia-docker2
   7. sudo pkill -SIGHUP dockerd

   #### Test nvidia-smi with the latest official CUDA image,here we specifically install install nvidia -driver 

   8. docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi

   ##### **cuda redirection: this is a necessary path in aws specially otherwise you'll get cuda compatible error**

   1. echo 'export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
   2. echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
   3. source ~/.bashrc

   ##### **Installing object detection docker**

   9. docker load -i od_tensorflow-latest.tar

---



1. nvidia-docker run  -p 0.0.0.0:6006:6006  -it  --rm -v /mnt/datastore/groups/ai/Mayank_datastore/im_work/dock_ob:/tensorflow 7d bash 

2. **docker run  -p 0.0.0.0:6006:6006 -it --gpus all  --name object_detect_tensorflow_api -v /home/mayank_sati/codebase/docker_ob:/tensorflow 7d6 bash** 

   

   Note: above **7d** is images id, it will be different for diff docker

3. cd ..

4. cd tensorflow/model/research

5. ```
   export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
   ```

- **for training the model**

CUSTOM CONFIG : data/config_files/faster_rcnn_resnet50_coco_custom_config.config

5. python object_detection/model_main.py \
       --pipeline_config_path=/tensorflow/data/config_files/faster_rcnn_resnet50_coco.config \
       --model_dir=/tensorflow/ckpt_files \
       --num_train_steps=100000\
       --sample_1_of_n_eval_examples=1 \
       --alsologtostderr

6. Inference and conversion is similar to previous training steps

7. python object_detection/export_inference_graph.py \
       --input_type=image_tensor \
       --pipeline_config_path=/tensorflow/data/config_files/faster_rcnn_resnet50_coco.config \
       --trained_checkpoint_prefix=/tensorflow/ckpt_files \
       --output_directory=/tensorflow/output

8. after training is complete:

9. tensorboard: (this can be run on same docke or create another docker container)

   1. tensorboard --logdir=/tensorflow/ckpt_files
   2. Go to link localhost : <http://localhost:6006/

   ----

   

cd /tensorflow/models/research



tensorboard --logdir=/tensorflow/ckpt_files

tensorboard --logdir=/tensorflow/ckpt

/mnt/datastore/groups/ai/Mayank_datastore/ckpt



```bash
docker run -it --rm -p 8888:8888 lspvic/tensorboard-notebook
```

```bsh
docker run -it --rm 7d6 \
   python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```

  

```bsh
docker run -it --rm tensorflow/tensorflow \
   python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```

```
-p 0.0.0.0:6006:6006
```

docker run -it --rm -p 0.0.0.0:6006:6006 77c6988d0d3a bash

----

this step are for tensorboard

nvidia-docker run  -p 0.0.0.0:6006:6006  -it  --rm -v /mnt/datastore/groups/ai/Mayank_datastore/im_work/dock_ob:/tensorflow 7d bash 



Go to link localhost : <http://localhost:6006/>

Enjoy tensorboard.

docker run -it --gpus all  --name object_detect_tensorflow_api -v /home/mayank_sati/codebase/docker_ob:/tensorflow d67 bash 

tensorboard --logdir=/home/mayank_sati/codebase/docker_ob/ckpt_files

----

yolo

1. To check fps

   1. guassian yolo

      ```
      ./darknet detector demo cfg/BDD.data cfg/Gaussian_yolov3_BDD.cfg Gaussian_yolov3_BDD.weights data/farm_full_image.avi -benchmark
      ```

   2. yolo v3

      ```
      ./darknet detector demo cfg/coco.data  cfg/yolov3.cfg  yolov3.weights  data/farm_full_image.avi -benchmark
      ```

      

   3. yolov4

      ```
      ./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights  data/farm_full_image.avi -benchmark
      
      ./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights data/farminigton_area.avi  -ext_output
      ```

      

   4. -thresh 0.25

   5. Genrate result for evaluation.

      .

      ```
      /darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result_yolov4.json < data/train.txt
      
      ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.txt < data/train.tx
      
      
      ```

   6. Creating evalulation data 

      ```
      ./darknet detector test cfg/BDD.data cfg/Gaussian_yolov3_BDD.cfg Gaussian_yolov3_BDD.weights -dont_show -ext_output < data/val_bdd_list_modified.txt > result_gaussian_bdd_2.txt
      
      
      ./darknet detector test cfg/BDD.data cfg/yolov4.cfg yolov4.weights -dont_show -ext_output < data/val_bdd_list_modified.txt > result_yolov4_bdd_2.txt
      
      ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < data/train.txt
      
      ```

   7. dfs

   8. 

