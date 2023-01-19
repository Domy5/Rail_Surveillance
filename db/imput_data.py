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
    
    camera_1 = controler.Camera(camera_id = '1', line = 'L06', station = 'Puerta del Ángel', platform = '2' , against_direction = '0', path = 'video_test/01.mp4') 
    camera_2 = controler.Camera(camera_id = '2', line = 'L05', station = 'Marqués de Vadillo', platform = '2' , against_direction = '0', path = 'video_test/02.mp4') 
    camera_3 = controler.Camera(camera_id = '3', line = 'L03', station = 'Lavapiés', platform = '2' , against_direction = '0', path = 'video_test/03.mp4')
    camera_4 = controler.Camera(camera_id = '4', line = 'L01', station = 'Estrecho', platform = '1' , against_direction = '0', path = 'video_test/04.mp4')        
    camera_5 = controler.Camera(camera_id = '5', line = 'L05', station = 'Chueca', platform = '1' , against_direction = '0', path = 'video_test/05.mp4')
    camera_6 = controler.Camera(camera_id = '6', line = 'L02', station = 'Opera', platform = '2' , against_direction = '0', path = 'video_test/06.mp4')
    camera_7 = controler.Camera(camera_id = '7', line = 'L01', station = 'Sol', platform = '1' , against_direction = '0', path = 'video_test/07.mp4')
    
    camera_8 = controler.Camera(camera_id = '8', line = 'L06', station = 'Puerta del Ángel', platform = '2' , against_direction = '0', path = 'video_test/01 b.mp4')
    camera_9 = controler.Camera(camera_id = '9', line = 'L05', station = 'Marqués de Vadillo', platform = '2' , against_direction = '0', path = 'video_test/02 b.mp4')
    camera_10 = controler.Camera(camera_id = '10', line = 'L05', station = 'Chueca', platform = '1' , against_direction = '0', path = 'video_test/05 b.mp4')
    camera_11 = controler.Camera(camera_id = '11', line = 'L02', station = 'Opera', platform = '1' , against_direction = '0', path = 'video_test/06 b.mp4')
    camera_12 = controler.Camera(camera_id = '12', line = 'L01', station = 'Sol', platform = '1' , against_direction = '0', path = 'video_test/07 b.mp4')

    # # # # # # # # # # # # # # # # 
    #                             #
    #       (|,__)                #
    #    1  (point11,point12)     #
    #    2  (point21,point22)     #
    #    3  (point31,point32)     #
    #    4  (point41,point42)     #
    #    5  (point51,point52)     #
    #                             #
    # # # # # # # # # # # # # # # # 
    # (0,0)       ___________     # (640,0)
    #         2  *          * 3   #
    #           *          *      #
    #          *          *       #
    #         *          *        #
    #        *          *         #
    #       *          *          #
    #      *          *           #
    #   1 *          *            #
    #     |         *             #
    #     |        *              #
    #     |       *               #
    #   5 |______* 4              #
    #                             # 
    # (0,480)                     # (640,480)
    # # # # # # # # # # # # # # # #
    
    ROI_polygon_1 = controler.ROI_polygon(polygon_id = '1', point_11 = '0', point_12 = '195', point_21 = '350', point_22 = '0', point_31 = '384', point_32 = '0', point_41 = '252', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '1') 
    ROI_polygon_2 = controler.ROI_polygon(polygon_id = '2', point_11 = '0', point_12 = '155', point_21 = '228', point_22 = '0', point_31 = '246', point_32 = '0', point_41 = '181', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '2')
    ROI_polygon_3 = controler.ROI_polygon(polygon_id = '3', point_11 = '0', point_12 = '235', point_21 = '388', point_22 = '98', point_31 = '480', point_32 = '100', point_41 = '198', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '3')
    ROI_polygon_4 = controler.ROI_polygon(polygon_id = '4', point_11 = '0', point_12 = '240', point_21 = '400', point_22 = '80', point_31 = '493', point_32 = '80', point_41 = '290', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '4')
    ROI_polygon_5 = controler.ROI_polygon(polygon_id = '5', point_11 = '0', point_12 = '220', point_21 = '276', point_22 = '0', point_31 = '295', point_32 = '0', point_41 = '146', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '5')
    ROI_polygon_6 = controler.ROI_polygon(polygon_id = '6', point_11 = '0', point_12 = '90', point_21 = '194', point_22 = '15', point_31 = '278', point_32 = '15', point_41 = '142', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '6')
    ROI_polygon_7 = controler.ROI_polygon(polygon_id = '7', point_11 = '0', point_12 = '140', point_21 = '274', point_22 = '30', point_31 = '321', point_32 = '30', point_41 = '90', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '7')
    
    ROI_polygon_8 = controler.ROI_polygon(polygon_id = '8', point_11 = '0', point_12 = '195', point_21 = '350', point_22 = '0', point_31 = '375', point_32 = '0', point_41 = '240', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '8')
    ROI_polygon_9 = controler.ROI_polygon(polygon_id = '9', point_11 = '0', point_12 = '155', point_21 = '228', point_22 = '0', point_31 = '235', point_32 = '0', point_41 = '135', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '9')
    ROI_polygon_10 = controler.ROI_polygon(polygon_id = '10', point_11 = '0', point_12 = '185', point_21 = '261', point_22 = '0', point_31 = '342', point_32 = '0', point_41 = '192', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '10')
    ROI_polygon_11 = controler.ROI_polygon(polygon_id = '11', point_11 = '0', point_12 = '132', point_21 = '43', point_22 = '98', point_31 = '50', point_32 = '43', point_41 = '121', point_42 = '43', point_51 = '129', point_52 = '478', camera_id = '11')
    ROI_polygon_12 = controler.ROI_polygon(polygon_id = '12', point_11 = '0', point_12 = '169', point_21 = '174', point_22 = '69', point_31 = '254', point_32 = '69', point_41 = '100', point_42 = '480', point_51 = '0', point_52 = '480', camera_id = '12')
    
    
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
   
    session.commit()
    session.close()
    
except exc.IntegrityError as e:
    print ('Error: Integrity')
    print (e)
except exc.SQLAlchemyError as e:
    print ('Error: Integrity')
    print (e)
    session.close()