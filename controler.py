#!/usr/bin/env python
# coding: utf-8

# # https://www.youtube.com/watch?v=XSAjQDM8ZS4
# https://www.youtube.com/watch?v=uB0928SOTEQ
# https://www.sqlalchemy.org/
# 
# #

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
Base = declarative_base()

class Camara(Base):
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

class ROI_poligono(Base):
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
    
Session = sessionmaker(engine)
session = Session()

if __name__ == "__main__": 
    Base.metadata.drop_all(engine) 
    Base.metadata.create_all(engine)
    
    try:
  
        camara1 = Camara(id_camara = '1', linea = 'L01', estacion = 'Chamartin', anden = '1' , contramarcha = '0', ruta = 'test_video/a1-003 1 minuto 1 via.mkv') 
        camara2 = Camara(id_camara = '2', linea = 'L02', estacion = 'Las Rosas', anden = '1' , contramarcha = '0', ruta = 'test_video/output.avi') 
        camara3 = Camara(id_camara = '3', linea = 'L03', estacion = 'Las ', anden = '1' , contramarcha = '0', ruta = 'test_video/Dramatic_footage_Woman_falls_on_Madrid039s_Metro_t.mp4')
        camara4 = Camara(id_camara = '4', linea = 'L03', estacion = 'Las ', anden = '1' , contramarcha = '0', ruta = 'test_video/Cada_en_el_Metro_de_Madrid2.mp4')        
        camara5 = Camara(id_camara = '5', linea = 'L03', estacion = 'Las ', anden = '1' , contramarcha = '0', ruta = 'test_video/Rescatada justo antes de ser arrollada por el metro de Madrid.mp4')
        camara6 = Camara(id_camara = '6', linea = 'L03', estacion = 'Las ', anden = '1' , contramarcha = '0', ruta = 'test_video/Rescatada justo antes de ser arrollada por el metro de Madrid (1).mp4')
      
    #area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]]) 
        ROI_poligono1 = ROI_poligono(id_poligono = '1', punto11 = '0', punto12 = '195', punto21 = '350', punto22 = '0', punto31 = '384', punto32 = '0', punto41 = '252', punto42 = '480', punto51 = '0', punto52 = '480', id_camara = '1') 
        ROI_poligono2 = ROI_poligono(id_poligono = '2', punto11 = '0', punto12 = '155', punto21 = '228', punto22 = '0', punto31 = '257', punto32 = '0', punto41 = '195', punto42 = '478', punto51 = '0', punto52 = '477', id_camara = '2')
        ROI_poligono3 = ROI_poligono(id_poligono = '3', punto11 = '0', punto12 = '201', punto21 = '378', punto22 = '0', punto31 = '478', punto32 = '0', punto41 = '365', punto42 = '718', punto51 = '0', punto52 = '718', id_camara = '3')
        ROI_poligono4 = ROI_poligono(id_poligono = '4', punto11 = '0', punto12 = '201', punto21 = '378', punto22 = '0', punto31 = '478', punto32 = '0', punto41 = '365', punto42 = '718', punto51 = '0', punto52 = '718', id_camara = '4')
        ROI_poligono5 = ROI_poligono(id_poligono = '5', punto11 = '0', punto12 = '136', punto21 = '214', punto22 = '0', punto31 = '248', punto32 = '0', punto41 = '196', punto42 = '357', punto51 = '0', punto52 = '357', id_camara = '5')
        ROI_poligono6 = ROI_poligono(id_poligono = '6', punto11 = '0', punto12 = '136', punto21 = '214', punto22 = '0', punto31 = '248', punto32 = '0', punto41 = '196', punto42 = '357', punto51 = '0', punto52 = '357', id_camara = '6')
        
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
    
    #return poligono[0]
    return poligono

############### pruebas

ruta = get_ruta_video(3)

print(ruta)

poli = get_ROI_video(3)

print("tipo: " + str(type(poli)))

print(poli)




