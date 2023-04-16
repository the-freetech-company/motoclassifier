# docs - https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection

from imageai.Detection import ObjectDetection
import os
import shutil
        

def extract_objs_from_images(input_dir: str, output_dir: str) -> None:
    '''
    input_dir: Input directory containing folders with unprocessed input.
    output dir: Output dir where folders containing proccessed input will be placed.
    
    1. Loops through every folder in the input dir and every image within each folder.
    2. This method then identifies motorbike and person within each image.
    3. After identification, this method will crop the objects identified and save them in the output dir, respective to its containing folder.
    '''
    # Arrange
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolov3.pt"))
    custom_objects = detector.CustomObjects(person=True, motorbike=True)
    detector.loadModel()
    
    
    # Act
    folders = os.listdir(input_dir)
    for folder in folders[2:]:
        images = os.listdir('{0}/{1}'.format(input_dir, folder))
        os.makedirs(os.path.join(execution_path, output_dir, folder), exist_ok=True)
        for image in images:
            image_path = '{0}/{1}/{2}'.format(input_dir, folder, image)
            output_image_path = os.path.join(execution_path ,'{0}/{1}'.format(output_dir,folder))
            output_image = f'{output_image_path}/{image}_detected.jpg'
            detections, extracted_objects  = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , image_path),
                                                                            extract_detected_objects=True,
                                                                            output_image_path=output_image,
                                                                            custom_objects=custom_objects,
                                                                            minimum_percentage_probability=80)
            os.remove(output_image)            
            for i, obj in enumerate(detections):
                print(obj)
                new_image_name = '{0}-{1}{2}'.format(image.replace('.jpg', ''), detections[i]['name'], '.jpg')
                new_path = '{0}/{1}'.format(output_image_path, new_image_name)
                shutil.copy(extracted_objects[i], new_path) 
            if extracted_objects:
                shutil.rmtree(os.path.dirname(extracted_objects[0]))
                    

extract_objs_from_images('./bikes', 'data')