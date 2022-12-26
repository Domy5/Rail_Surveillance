#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Domingo Martínez Núñez
# Created Date: abril 2022
# version ='1.0'
# https://github.com/Domy5/Rail_Surveillance/
# ---------------------------------------------------------------------------
""" Detention of people and/or objects on the railway platform in real time """
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
engine = create_engine('sqlite:///db/Cameras.sqlite')
base = declarative_base()

class Camera(base):
    __tablename__ = 'camera'

    camera_id = Column(Integer(), primary_key=True)
    line = Column(String(3), nullable=False, unique=False)
    station = Column(String(50), nullable=False, unique=False)
    platform = Column(Integer(), nullable=False, unique=False)
    against_direction = Column(Integer(), nullable=False, unique=False)
    path = Column(String(100), nullable=True, unique=True)
    ROI_polygon = relationship('ROI_polygon')
    
    def __str__(self):
        return self.camera_id

class ROI_polygon(base):
    __tablename__ = 'ROI_polygon'
    
    polygon_id = Column(Integer, primary_key=True)
    point_11	= Column(Integer, nullable=False, unique=False)
    point_12	= Column(Integer, nullable=False, unique=False)
    point_21	= Column(Integer, nullable=False, unique=False)
    point_22	= Column(Integer, nullable=False, unique=False)
    point_31	= Column(Integer, nullable=False, unique=False)
    point_32	= Column(Integer, nullable=False, unique=False)
    point_41	= Column(Integer, nullable=False, unique=False)
    point_42	= Column(Integer, nullable=False, unique=False)
    point_51	= Column(Integer, nullable=False, unique=False)
    point_52	= Column(Integer, nullable=False, unique=False)
    camera_id = Column(Integer, ForeignKey("camera.camera_id"))

    def __str__(self):
        return self.polygon_id

def get_video_path(camera_id):
    
    Session = sessionmaker(bind=engine) 
    session = Session() 
    
    camera = session.query(Camera).filter(Camera.camera_id == camera_id).all()
    
    session.close()
    
    return camera[0].path

def get_video_ROI(camera_id):
    
    Session = sessionmaker(bind=engine) 
    session = Session() 
    
    polygon = session.query(ROI_polygon).filter(ROI_polygon.camera_id == camera_id).all()
    
    session.close()
    
    # area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])

    polygon = [[polygon[0].point_11,polygon[0].point_12],[polygon[0].point_21,polygon[0].point_22],[polygon[0].point_31,polygon[0].point_32],[polygon[0].point_41,polygon[0].point_42],[polygon[0].point_51,polygon[0].point_52]]
    
    return polygon
