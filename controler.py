# https://www.youtube.com/watch?v=XSAjQDM8ZS4
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

from pruebas import Punto22



engine = create_engine('sqlite:///BBDD/Camaras.sqlite', echo=True)
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

    camara1 = Camara(id_camara = '1', linea = 'L01', estacion = 'Chamartin', anden = '1' , contramarcha = '0', ruta = 'videos_pruebas/a1-003 1 minuto 3 via.mkv')
    
    #area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])
    ROI_poligono1 = ROI_poligono(id_poligono = '1', punto11 = '0', punto12 = '195', punto21 = '350', punto22 = '0', punto31 = '384', punto32 = '0', punto41 = '252', punto42 = '480', punto51 = '0', punto52 = '480', id_camara = '1')
    
    session.add(camara1)
    session.add(ROI_poligono1)
    
    session.commit()