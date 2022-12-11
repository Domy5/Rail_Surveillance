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
# https://www.sqlalchemy.org/


import sqlite3 as sql
import sqlalchemy 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Table, ForeignKey
from sqlalchemy import create_engine

from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError as exc


# engine = create_engine('sqlite:///db/Camaras.sqlite', echo=True)
engine = create_engine('sqlite:///db/Camaras.sqlite')
base = declarative_base()

class Camara(base):
    __tablename__ = 'camara'

    id_camara = Column(Integer(), primary_key=True)
    linea = Column(String(3), nullable=False, unique=False) #L10
    estacion = Column(String(50), nullable=False, unique=False)
    anden = Column(Integer(), nullable=False, unique=False)
    contramarcha = Column(Integer(), nullable=False, unique=False)
    ruta = Column(String(100), nullable=True, unique=True)
    ROI_poligono = relationship('ROI_poligono')
    
    def __str__(self):
        return self.id_camara

class ROI_poligono(base):
    __tablename__ = 'ROI_poligono'
    
    id_poligono = Column(Integer, primary_key=True)
    punto11	= Column(Integer, nullable=False, unique=False)
    punto12	= Column(Integer, nullable=False, unique=False)
    punto21	= Column(Integer, nullable=False, unique=False)
    punto22	= Column(Integer, nullable=False, unique=False)
    punto31	= Column(Integer, nullable=False, unique=False)
    punto32	= Column(Integer, nullable=False, unique=False)
    punto41	= Column(Integer, nullable=False, unique=False)
    punto42	= Column(Integer, nullable=False, unique=False)
    punto51	= Column(Integer, nullable=False, unique=False)
    punto52	= Column(Integer, nullable=False, unique=False)
    punto_medio11	= Column(Integer, nullable=True, unique=False)
    punto_medio12	= Column(Integer, nullable=True, unique=False)
    punto_medio21	= Column(Integer, nullable=True, unique=False)
    punto_medio22	= Column(Integer, nullable=True, unique=False)
    id_camara = Column(Integer, ForeignKey("camara.id_camara"))

    def __str__(self):
        return self.id_poligono
    


def get_ruta_video(id_camara):
    
    Session = sessionmaker(bind=engine) 
    session = Session() 
    
    camara = session.query(Camara).filter(Camara.id_camara == id_camara).all()
    
    session.close()
    
    return camara[0].ruta

def get_ROI_video(id_camara):
    
    Session = sessionmaker(bind=engine) 
    session = Session() 
    
    poligono = session.query(ROI_poligono).filter(ROI_poligono.id_camara == id_camara).all()
    
    session.close()
    
    # area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])

    poligono = [[poligono[0].punto11,poligono[0].punto12],[poligono[0].punto21,poligono[0].punto22],[poligono[0].punto31,poligono[0].punto32],[poligono[0].punto41,poligono[0].punto42],[poligono[0].punto51,poligono[0].punto52]]
    
    return poligono
