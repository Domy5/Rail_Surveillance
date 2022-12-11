#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Domingo Martínez Núñez
# Created Date: abril 2022
# version ='1.0'
# https://github.com/Domy5/Rail_Surveillance/
# ---------------------------------------------------------------------------
""" Deteccion de objetos en plataforma de vias ferroviarias """
# ---------------------------------------------------------------------------

import sqlite3 as sql
import sqlalchemy 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Table, ForeignKey
from sqlalchemy import create_engine

from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError as exc

import controler

Session = sessionmaker(controler.engine)
session = Session()

#if __name__ == "__main__": 
controler.Base.metadata.drop_all(controler.engine) 
controler.Base.metadata.create_all(controler.engine)

try:

    camara1 = controler.Camara(id_camara = '1', linea = 'L01', estacion = 'Chamartin', anden = '1' , contramarcha = '0', ruta = 'test_video/a1-003 1 minuto 1 via.mkv') 
    camara2 = controler.Camara(id_camara = '2', linea = 'L02', estacion = 'Las Rosas', anden = '1' , contramarcha = '0', ruta = 'test_video/output.avi') 
    camara3 = controler.Camara(id_camara = '3', linea = 'L03', estacion = 'Las ', anden = '1' , contramarcha = '0', ruta = 'test_video/Dramatic_footage_Woman_falls_on_Madrid039s_Metro_t.mp4')
    camara4 = controler.Camara(id_camara = '4', linea = 'L03', estacion = 'Las ', anden = '1' , contramarcha = '0', ruta = 'test_video/Cada_en_el_Metro_de_Madrid2.mp4')        
    camara5 = controler.Camara(id_camara = '5', linea = 'L03', estacion = 'Las ', anden = '1' , contramarcha = '0', ruta = 'test_video/Rescatada justo antes de ser arrollada por el metro de Madrid.mp4')
    camara6 = controler.Camara(id_camara = '6', linea = 'L03', estacion = 'Las ', anden = '1' , contramarcha = '0', ruta = 'test_video/Rescatada justo antes de ser arrollada por el metro de Madrid (1).mp4')
  
#area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]]) 
    ROI_poligono1 = controler.ROI_poligono(id_poligono = '1', punto11 = '0', punto12 = '195', punto21 = '350', punto22 = '0', punto31 = '384', punto32 = '0', punto41 = '252', punto42 = '480', punto51 = '0', punto52 = '480', id_camara = '1') 
    ROI_poligono2 = controler.ROI_poligono(id_poligono = '2', punto11 = '0', punto12 = '155', punto21 = '228', punto22 = '0', punto31 = '257', punto32 = '0', punto41 = '195', punto42 = '478', punto51 = '0', punto52 = '477', id_camara = '2')
    ROI_poligono3 = controler.ROI_poligono(id_poligono = '3', punto11 = '0', punto12 = '201', punto21 = '378', punto22 = '0', punto31 = '478', punto32 = '0', punto41 = '365', punto42 = '718', punto51 = '0', punto52 = '718', id_camara = '3')
    ROI_poligono4 = controler.ROI_poligono(id_poligono = '4', punto11 = '0', punto12 = '201', punto21 = '378', punto22 = '0', punto31 = '478', punto32 = '0', punto41 = '365', punto42 = '718', punto51 = '0', punto52 = '718', id_camara = '4')
    ROI_poligono5 = controler.ROI_poligono(id_poligono = '5', punto11 = '0', punto12 = '136', punto21 = '214', punto22 = '0', punto31 = '248', punto32 = '0', punto41 = '196', punto42 = '357', punto51 = '0', punto52 = '357', id_camara = '5')
    ROI_poligono6 = controler.ROI_poligono(id_poligono = '6', punto11 = '0', punto12 = '136', punto21 = '214', punto22 = '0', punto31 = '248', punto32 = '0', punto41 = '196', punto42 = '357', punto51 = '0', punto52 = '357', id_camara = '6')
    
    session.add(camara1) 
    session.add(camara2)
    session.add(camara3)
    session.add(camara4)
    session.add(camara5)
    session.add(camara6)
    
    session.add(ROI_poligono1)  
    session.add(ROI_poligono2) 
    session.add(ROI_poligono3) 
    session.add(ROI_poligono4) 
    session.add(ROI_poligono5) 
    session.add(ROI_poligono6) 
  
    session.commit()
    session.close()
    
except exc.IntegrityError as e:
    print ('Error: Integrity')
    print (e)
except exc.SQLAlchemyError as e:
    print ('Error: Integrity')
    print (e)
    session.close()