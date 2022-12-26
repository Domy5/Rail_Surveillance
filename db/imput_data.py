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

import sqlite3 as sql
import sqlalchemy 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError as exc

import controler

Session = sessionmaker(controler.engine)
session = Session()

#if __name__ == "__main__": 
controler.base.metadata.drop_all(controler.engine) 
controler.base.metadata.create_all(controler.engine)

try:

    camera_1 = controler.Camera(camera_id = '1', line = 'L01', station = 'Chamartin', platform = '1' , against_direction = '0', path = 'video_test/a1-003 1 minuto 1 via.mkv') 
    camera_2 = controler.Camera(camera_id = '2', line = 'L02', station = 'Las Rosas', platform = '1' , against_direction = '0', path = 'video_test/output.avi') 
    camera_3 = controler.Camera(camera_id = '3', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/Dramatic_footage_Woman_falls_on_Madrid039s_Metro_t.mp4')
    camera_4 = controler.Camera(camera_id = '4', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/Cada_en_el_Metro_de_Madrid2.mp4')        
    camera_5 = controler.Camera(camera_id = '5', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/Rescatada justo antes de ser arrollada por el metro de Madrid.mp4')
    camera_6 = controler.Camera(camera_id = '6', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/Rescatada justo antes de ser arrollada por el metro de Madrid (1).mp4')
  
    # area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])
    
    ROI_polygon_1 = controler.ROI_polygon(polygon_id = '1', point_11 = '0', point_12 = '195', point_21 = '350', point_22 = '0', point_31 = '384', point_32 = '0', point_41 = '252', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '1') 
    ROI_polygon_2 = controler.ROI_polygon(polygon_id = '2', point_11 = '0', point_12 = '155', point_21 = '228', point_22 = '0', point_31 = '257', point_32 = '0', point_41 = '195', point_42 = '478', point_51 = '0', point_52 = '477', camera_id = '2')
    ROI_polygon_3 = controler.ROI_polygon(polygon_id = '3', point_11 = '0', point_12 = '201', point_21 = '378', point_22 = '0', point_31 = '478', point_32 = '0', point_41 = '365', point_42 = '718', point_51 = '0', point_52 = '718', camera_id = '3')
    ROI_polygon_4 = controler.ROI_polygon(polygon_id = '4', point_11 = '0', point_12 = '201', point_21 = '378', point_22 = '0', point_31 = '478', point_32 = '0', point_41 = '365', point_42 = '718', point_51 = '0', point_52 = '718', camera_id = '4')
    ROI_polygon_5 = controler.ROI_polygon(polygon_id = '5', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '5')
    ROI_polygon_6 = controler.ROI_polygon(polygon_id = '6', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '6')
    
    session.add(camera_1) 
    session.add(camera_2)
    session.add(camera_3)
    session.add(camera_4)
    session.add(camera_5)
    session.add(camera_6)
    
    session.add(ROI_polygon_1)  
    session.add(ROI_polygon_2) 
    session.add(ROI_polygon_3) 
    session.add(ROI_polygon_4) 
    session.add(ROI_polygon_5) 
    session.add(ROI_polygon_6) 
  
    session.commit()
    session.close()
    
except exc.IntegrityError as e:
    print ('Error: Integrity')
    print (e)
except exc.SQLAlchemyError as e:
    print ('Error: Integrity')
    print (e)
    session.close()