ici je doit deja expliqué les installation necessaire

l'environnement conda tensorflow et les dependance et commenet jai créé mon environement python pour tensorflow gpu

cnn.py : jai fait un script pour entrainer et sortir un model capable de reconnaitre 65 fruit : model.h5

app.py : dedans un serveur web pour testé la reconnaissance dde fruit selon le model entrainné dans le cnn.py un upload dimage puis ca compare avec ce quil a appris et predit le resultat

cnn-new-more-fruit : un autre reseau de nouronne capable d'entrainer un model a reconnaitre plus de fruit ici jai mis 65 classes


quand je rajoute des data entrainement et test il faut changé le path si besoin , les classes dans le app ou dans le script qui lance la video et faut surtout changé le nombre de neuronne en sortie
da ns la fonction qui ajoute les dense faut changé le parame unit dans la ligne ou on fait le softmax au nombre de sortie qui est le nombre de classes


les script detection et detection_tf2 concerne la reconnaissance d'objet en live via camera les deux script utilise des model preentrainer que jai telechargé d'internet sur le github suivant :
 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
 je compte men servir pour faire fct la cam avec les fruit, la detection fonctonne bien avec le model ssd du net
 sauf que il utilise un frozen graph en plus donc mon model seule suffit pas a lancé le truc a voir


