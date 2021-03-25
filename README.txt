ici je doit deja expliqué les installation necessaire
/////////////////////////////////////////////////////////

veuillez verifiez d'abord la compatibilté de votre gpu avec tensorflow sur : https://www.tensorflow.org/install/gpu

dans le cas ou votre gpu n'est pas compatible vous aurez qu'a utilisé tensorflow qui va faire les calcul sur CPU chose qui prendrais bcp de temps mais cela fonctionnera quand meme

pour l'environement j'ai utilisé :
python 3.8
tensorflow 2.4 avec support gpu
cuda 11.2
cuDNN 8

1 : j'ai utlisé visual studio pour installé CUDA Toolkit
2 : installer CUDA Toolkit
3: telechargez et installer cuDNN : pour cela telecharger cuDNN sur le site de nvidia
    dezzipez le puis coupez et coller les fichier contenu dans cuDNN dans le dossier qui contient CUDA ( je tacherais de mieux l'expliquer)
4 :  n'oubliez pas de mettre les path dans les parametre d'environnement si ce n'est pas fait automatiquement a l'installation de cuda et de cudnn
5 : installez python
6 : installez Tensorflow avec support GPU avec la commande en mode admin : pip3 install --upgrade tensorflow-gpu
testez tensorflow avec la commande : import tensorflow as tf
testez cuda avec la commande : tf.test.is_built_with_cuda()
testez la compatibility gpu avec tensorflow avec la commande : tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

///////////////////////////////////////////////////////////////

l'environnement conda tensorflow et les dependance et commenet jai créé mon environement python pour tensorflow gpu

cnn.py : jai fait un script pour entrainer et sortir un model capable de reconnaitre 65 fruit : model.h5

app.py : dedans un serveur web pour testé la reconnaissance dde fruit selon le model entrainné dans le cnn.py un upload dimage puis ca compare avec ce quil a appris et predit le resultat

cnn-new-more-fruit : un autre reseau de nouronne capable d'entrainer un model a reconnaitre plus de fruit ici jai mis 65 classes


quand je rajoute des data entrainement et test il faut changé le path si besoin , les classes dans le app ou dans le script qui lance la video et faut surtout changé le nombre de neuronne en sortie
da ns la fonction qui ajoute les dense faut changé le parame unit dans la ligne ou on fait le softmax au nombre de sortie qui est le nombre de classes

////////////////////////////////////////////////

pour entrainer un model va falloir d'abord des data, j'en ai push sur un repository un dataset contenant deux deossier entrainement et validation "test" le repos est visible si besoin de telechargé les data :

https://github.com/hichemseriket/reconnaissance_de_fruit/tree/camera-fruit

le dossier des data s'appel Fruit-images-Dataset, vous pouvez egalement mettre n'importe quel dataset faut juste bien separé et surtt bien les appele training et validation comme dans le script cnn ,
sinon changez les nommage dans le code

pour faire fonctionner la reconnaissance de fruit faut lancé le : cnn.py , qui va entrainer un model et le créé sous le nom de model.h5

ensuite vous lancer le serveur web flask : app.py

vous y accedez puis vous upload des image via leinput "choisir image" puis vous cliquez sur predire et cela va predire

faut bien que le fruit soit l'un des fruit que vous avez utiliser pour entrainer le model sinon il saura pas predire au mieux il predit le vpoisin le plus proche.
exemple: si on veut predire une orange alors que notre model ne connais pas orange par contre il conais bien pomme il dira que cest une pomme,
en fait le label inconnu n'existe pas on peut le rajouté en vrai cela evitera de predire le voisin lme plus proche

il faut bien aussi que les classe dans le app qui corresponde au fruit de lentrainement soit les meme que les fruit de trai et de test donc les classe au meme nombre que les labels "nom des fichier "


///////////////////////////////////////////////////////

les script detection et detection_tf2 concerne la reconnaissance d'objet en live via camera les deux script utilise des model preentrainer que j'ai telechargé d'internet sur le github suivant :

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

je compte m'en servir pour faire fct la cam avec les fruit, la detection fonctonne bien avec le model ssd du net
sauf que il utilise un frozen graph en plus donc mon model seule suffit pas a lancé le truc a voir

pour lancer le script suffit de lancer detection.py ou detection-tf2.py ( je vous conseil le tf2 meme si detection qui concernais tf1 je lai normalement adapté a tf2 du coup les deux font la meme choise )
en vrai vu que jai chalngé toute les commande de tensorflow 1 dans detection j'ai finalement le meme script que detection-tf2 juste quelque noms de label"classes" qui change

pour pouvoir l'utlisez veuillez bien telechargé un model preentrainer depuis le git au dessus et bien faire attention au path.
moi je sors toujours le model ainsi que le frzen du dossier telechargé du sorte a pas avoir a ecrire des path

voila la camera de votre pc va se lancer et va se mettre a predire les objets qu'elle vois