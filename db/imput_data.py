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
    
    camera_1 = controler.Camera(camera_id = '1', line = 'L01', station = 'Chamartin', platform = '1' , against_direction = '0', path = 'video_test/1.avi') 
    camera_2 = controler.Camera(camera_id = '2', line = 'L02', station = 'Las Rosas', platform = '1' , against_direction = '0', path = 'video_test/2.avi') 
    camera_3 = controler.Camera(camera_id = '3', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/3.mp4')
    camera_4 = controler.Camera(camera_id = '4', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/4.mp4')        
    camera_5 = controler.Camera(camera_id = '5', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/5.mp4')
    camera_6 = controler.Camera(camera_id = '6', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/6.mp4')
    camera_7 = controler.Camera(camera_id = '7', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/7.mp4')
    camera_8 = controler.Camera(camera_id = '8', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/8.mp4')
    camera_9 = controler.Camera(camera_id = '9', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/9.mp4')
    camera_10 = controler.Camera(camera_id = '10', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/10.mp4')
    camera_11 = controler.Camera(camera_id = '11', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/11.mp4')
    camera_12 = controler.Camera(camera_id = '12', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/12.mp4')
    camera_13 = controler.Camera(camera_id = '13', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/13.mp4')
    camera_14 = controler.Camera(camera_id = '14', line = 'L03', station = 'Las ', platform = '1' , against_direction = '0', path = 'video_test/14.mp4')
  
    # area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])
    
    ROI_polygon_1 = controler.ROI_polygon(polygon_id = '1', point_11 = '0', point_12 = '195', point_21 = '350', point_22 = '0', point_31 = '384', point_32 = '0', point_41 = '252', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '1') 
    ROI_polygon_2 = controler.ROI_polygon(polygon_id = '2', point_11 = '0', point_12 = '155', point_21 = '228', point_22 = '0', point_31 = '257', point_32 = '0', point_41 = '195', point_42 = '478', point_51 = '0', point_52 = '477', camera_id = '2')
    ROI_polygon_3 = controler.ROI_polygon(polygon_id = '3', point_11 = '0', point_12 = '201', point_21 = '378', point_22 = '0', point_31 = '478', point_32 = '0', point_41 = '365', point_42 = '718', point_51 = '0', point_52 = '718', camera_id = '3')
    ROI_polygon_4 = controler.ROI_polygon(polygon_id = '4', point_11 = '0', point_12 = '201', point_21 = '378', point_22 = '0', point_31 = '478', point_32 = '0', point_41 = '365', point_42 = '718', point_51 = '0', point_52 = '718', camera_id = '4')
    ROI_polygon_5 = controler.ROI_polygon(polygon_id = '5', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '5')
    ROI_polygon_6 = controler.ROI_polygon(polygon_id = '6', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '6')
    ROI_polygon_7 = controler.ROI_polygon(polygon_id = '7', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '7')
    ROI_polygon_8 = controler.ROI_polygon(polygon_id = '8', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '8')
    ROI_polygon_9 = controler.ROI_polygon(polygon_id = '9', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '9')
    ROI_polygon_10 = controler.ROI_polygon(polygon_id = '10', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '10')
    ROI_polygon_11 = controler.ROI_polygon(polygon_id = '11', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '11')
    ROI_polygon_12 = controler.ROI_polygon(polygon_id = '12', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '12')
    ROI_polygon_13 = controler.ROI_polygon(polygon_id = '13', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '13')
    ROI_polygon_14 = controler.ROI_polygon(polygon_id = '14', point_11 = '0', point_12 = '136', point_21 = '214', point_22 = '0', point_31 = '248', point_32 = '0', point_41 = '196', point_42 = '357', point_51 = '0', point_52 = '357', camera_id = '14')
    
    session.add(camera_1) 
    session.add(camera_2)
    session.add(camera_3)
    session.add(camera_4)
    session.add(camera_5)
    session.add(camera_6)
    session.add(camera_7)
    session.add(camera_8)
    session.add(camera_9)
    session.add(camera_10)
    session.add(camera_11)
    session.add(camera_12)
    session.add(camera_13)
    session.add(camera_14)
    
    session.add(ROI_polygon_1)  
    session.add(ROI_polygon_2) 
    session.add(ROI_polygon_3) 
    session.add(ROI_polygon_4) 
    session.add(ROI_polygon_5)
    session.add(ROI_polygon_6) 
    session.add(ROI_polygon_7) 
    session.add(ROI_polygon_8) 
    session.add(ROI_polygon_9) 
    session.add(ROI_polygon_10) 
    session.add(ROI_polygon_11) 
    session.add(ROI_polygon_12) 
    session.add(ROI_polygon_13) 
    session.add(ROI_polygon_14) 
  
    session.commit()
    session.close()
    
except exc.IntegrityError as e:
    print ('Error: Integrity')
    print (e)
except exc.SQLAlchemyError as e:
    print ('Error: Integrity')
    print (e)
    session.close()