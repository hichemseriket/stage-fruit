ici je doit deja expliqué les installation necessaire
l'environnement conda tensorflow et les dependance et commenet jai créé mon environement python pour tensorflow gpu
cnn.py : jai fait un script pour entrainer et sortir un model capable de differencie sur 3 fruit les fresh et les pourri, ca me sort le model : model.h5
app.py : dedans un serveur web pour testé la reconnaissance dde fruit selon le model entrainné dans le cnn.py
cnn-new-more-fruit : un autre reseau de nouronne capable d'entrainer un model a reconnaitre plus de fruit ici jai mis 65 classes
le dossier reconnaissance d objet contient du code pour activer la camera avec reconnaissance grace a un model preentrainé je compte men servir pour faire fct la cam avec les fruit
le detection fonctonne bien avec le model ssd du net
sauf que il utilise un frozen graph en plus donc mon model seule suffit pas a lancé le truc a voir

quand je rajoute des data entrainement et test il faut changé le path si besoin , les classes dans le app ou dans le script qui lance la video et faut surtout changé le nombre de neuronne en sortie
da  ns la fonction qui ajoute les dense faut changé le parame unit dans la ligne ou on fait le softmax au nombre de sortie qui est le nombre de classes