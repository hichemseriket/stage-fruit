# Importing keras libraries and packages
# on utilise la librairie keras car est very famous dans le deep learning models, keras contient deux classe la sequentel et la fonctionel
# on utilise sequentiel car adding layer step by step et la sortie du precedente couche et utilisé comme entré de la suivante


# j'ai rajouté 13 claasse et entrainer le model '


#doc pour comprendre les function utlisé dans ce tuto : https://keras.io/api/preprocessing/image/
from keras.models import Sequential
# mes 4 couches
#convulution
from keras.layers import Convolution2D
#pour applatir l'image
from keras.layers import Flatten
#dense represente la couche des neurones artificiel
from keras.layers import Dense
# le pooling pour encore rendre limage plus petite et les calcul plus rapide
from keras.layers import MaxPooling2D

# step1 Initializing CNN
# j'initialise mon objet ici mon objet est classifier
classifier = Sequential()

# step2 adding 1st Convolution layer and Pooling layer
# jutilise la methode add pour ajouter les couches les unes apres les autres
#dans convolution il ya bcp de parametre et dommage le 32 le son a coupé pour le premier 32 type de filtre pour extraire different feature , le (3,3) est la taille du filtre
#input shape parametre qui decide la taille des image que je vais introduire dans mon reseau le 64 x 64 largeur hauteur, et le 3 color scal rgb
#activation fonction , on va utilisé la non linearité entre les image et relu function est la meilleure des fonction pour la non linearité
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# la taille de polling metrix (2,2)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step3 adding 2nd convolution layer and polling layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step4 Flattening the layers
#je transform mon image en vecteur
classifier.add(Flatten())

# step5 Full_Connection
# je passe les data au reseau de neuronnes
# units represente le nombre de neuronnes dans le resau
#relu toujours pour la non linearité

classifier.add(Dense(units=32, activation='relu'))

classifier.add(Dense(units=64, activation='relu'))

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=256, activation='relu'))

classifier.add(Dense(units=256, activation='relu'))
# classifier.add(Dense(units=512, activation='relu'))

# le nombre de nouronnes est seulment de 6 car c'est le nombre de mes sorties et il est du au fait que jai que 3 fruit chacun d'entre eux  possede 2 category donc au final j'ai 6 classes
# la fonction activation ici est softmax car softmax est utilise pour categorical classification
# classifier.add(Dense(units=65, activation='softmax'))
# je recree un model que jentraine uniquement sur 12 classe trop de classe lui a fait perdre la boule
classifier.add(Dense(units=12, activation='softmax'))


# step6 Compiling CNN

# optimizer est utilisé pour optimisé notre training effecacité on utilise adam car il est adapté l'apprentisage
# loss similarly to softmax , utilisé calcul les erreur et les actuel result et les injecte a lenbtrainement pour que affiné et pdate l'entrainement pour que entrainer laccuracy sera mieux en gros cest chaud de le comprendre
#metrics on utilise accuracy as performance metrics en peut aussi utilisé looss
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# donc une fois les couche ajouté et la compilation faite on passe a la couche qui applati l'image et la transforme en vecteur pour quelle soit injecté dans un nouronnes '

# step7 Fitting CNN to images

#keras.processing est une bonne library
from keras.preprocessing.image import ImageDataGenerator
# jinitialise cette classe avec un objet ici mon objet est train_datagen pour entrainer les data cette classe possede des parametres :
# premier parametre est rescale pour rescal la matrix value dans le rang 1./255
# shear_range est utlisé Pour cisailler les images au hasard
# zoom_range Pour agrandir les images de manière aléatoire
# horizontal_flip pour retourner au hasard la moitié des images horizontalement

train_datagen = ImageDataGenerator(rescale=1. / 255,  # To rescaling the image in range of [0,1], fr : Pour redimensionner l'image dans une plage de [0,1]
                                   shear_range=0.2,  # To randomly shear the images : Pour cisailler les images au hasard
                                   zoom_range=0.2,  # To randomly zoom the images : Pour agrandir les images de manière aléatoire
                                   horizontal_flip=True)  # for randomly flipping half of the images horizontally : pour retourner au hasard la moitié des images horizontalement

# dans le test on ne fait que rescalé les images le reste des traitement etait fait a lentrainement
# Imagedatagenerator est une fonction qui permets de Generates a tf.data.Dataset from image files in a directory.
# ex de traitement si ma structure est : main_directory/
# ...class_a/
# ......a_image_1.jpg
# ......a_image_2.jpg
# ...class_b/
# ......b_image_1.jpg
# ......b_image_2.jpg
#Then calling image_dataset_from_directory(main_directory,
# labels='inferred') will return a tf.data.Dataset that yields batches of images from the subdirectories class_a and class_b,
# together with labels 0 and 1 (0 corresponding to class_a and 1 corresponding to class_b).
# du coup les label et les classes sont bien les nom des sous dossier
test_datagen = ImageDataGenerator(rescale=1. / 255)

print("\nTraining the data...\n")

# il ya une autre methode de cette classe qui est flow from directory nous permets si on ne veux pas que les labels soit ecrite separemment des autres fichiers
# ex dans notre cas on a les label qui sont les noms des dossier qui sont extrait les labels directement depuis les nom des dossier et les applique automatiquement aux images,
# faut juste que les folders soit dans le meme repertoire sans devoir envoyé le path donc juste jecris le nom du repertoire comme dans notre cas train est dans le meme repertoire
# 1 avec les data train les 13 classes
training_set = train_datagen.flow_from_directory('train',
#2 avec les data contenu dans fruit bref 65 classe
# training_set = train_datagen.flow_from_directory('Training65',
                                                 target_size=(64, 64), # pareill que lors de linput plus haut donc la largeur et la hauteur de limage
                                                 batch_size=16,  # Total no. of batches
                                                 # batch a precisé : il dit il parlera plustard dans la video pourquoi 12
                                                 class_mode='categorical')
# class_mode on va classifier 6 class avec car on sait quon va classé
# selon une reponse de overstackflow : Since you are passing class_mode='categorical' you dont have to manually convert the labels to one hot encoded vectors using to_categorical().
# The Generator will return labels as categorical.
#class_mode: One of "categorical", "binary", "sparse", "input", or None. Default: "categorical". Determines the type of label arrays that are returned: -
# "categorical" will be 2D one-hot encoded labels,
# - "binary" will be 1D binary labels, "sparse" will be 1D integer labels,
# - "input" will be images identical to input images (mainly used to work with autoencoders).
# - If None, no labels are returned (the generator will only yield batches of image data,
# which is useful to use with model.predict_generator()). Please note that in case of class_mode None, the data still needs to reside in a subdirectory of directory for it to work correctly.

test_set = test_datagen.flow_from_directory('test',
# test_set = test_datagen.flow_from_directory('Validation65',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='categorical')


# luc regard moi stp pourquoi les paramettre samples, nb epoch et nb val pose probleme du coup si je les garde le code compil pas mais si je les retire cela fonctionne mais jai pas la meme chose que sur la video du coup
classifier.fit_generator(training_set,
                         # steps_per_epoch= 1225,# Total training images
                         # jai retirer les nombre et les remplacer par la variable lent de mes objet entrain et test et cela semble fonctionné
                         steps_per_epoch= len(training_set),# Total training images
                         epochs = 20,# Total no. of epochs
                         validation_data=test_set,
                         validation_steps = len(test_set))  # Total testing images

# jai modifié les parametre qui ne sont plus a jour par des param depuis la doc et je regard le resultat,
# jai mis epoch a la place de nb_epoch, steps_per_epoch a la place de samples_per_epoch et validation_steps a la place nb_val_samples

# step8 saving model

# classifier.save("model.h5")
classifier.save("model13.h5")

# ici jai pui modifier le code et ajouetr mes classes a moi et cela fonctionne reste juste a voir dans les data bien mles reorganiser entre train et test