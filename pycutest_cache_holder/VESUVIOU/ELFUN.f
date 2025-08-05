      SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, NCALCF, ITYPEE, 
     *                   ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, 
     *                   ICALCF, LTYPEE, LSTAEV, LELVAR, LNTVAR, 
     *                   LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, 
     *                   LEPVLU, IFFLAG, IFSTAT )
      INTEGER NCALCF, IFFLAG, LTYPEE, LSTAEV, LELVAR, LNTVAR
      INTEGER LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, LEPVLU
      INTEGER IFSTAT
      INTEGER ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
      INTEGER INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
      INTEGER ICALCF(LCALCF)
      DOUBLE PRECISION FUVALS(LFVALU), XVALUE(LXVALU), EPVALU(LEPVLU)
C
C  Problem name : VESUVIOU  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION H     , C     , S     , X     , XMC   
      DOUBLE PRECISION R     , A     , E     , F     , DRDC  
      DOUBLE PRECISION DRDS  , DADC  , DADS  , DEDC  , DEDS  
      DOUBLE PRECISION D2RDCS, D2RDS2, D2ADC2, D2ADCS, D2ADS2
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : GAUSSIAN  
C
       H      = XVALUE(IELVAR(ILSTRT+     1))
       C      = XVALUE(IELVAR(ILSTRT+     2))
       S      = XVALUE(IELVAR(ILSTRT+     3))
       X      = EPVALU(IPSTRT+     1)
       XMC    = X - C                                    
       R      = XMC / S                                  
       A      = - 0.5 * R * R                            
       E      = EXP( A )                                 
       F      = H * E                                    
       DRDC   = - 1.0 / S                                
       DRDS   = - XMC / S ** 2                           
       DADC   = - R * DRDC                               
       DADS   = - R * DRDS                               
       DEDC   = E * DADC                                 
       DEDS   = E * DADS                                 
       D2RDCS = 1.0 / S ** 2                             
       D2RDS2 = 2.0 * XMC / S ** 3                       
       D2ADC2 = - DRDC ** 2                              
       D2ADCS = - ( DRDC * DRDS + R * D2RDCS )           
       D2ADS2 = - ( DRDS ** 2 + R * D2RDS2 )             
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= F                                        
       ELSE
        FUVALS(IGSTRT+     1)= E                                        
        FUVALS(IGSTRT+     2)= F * DADC                                 
        FUVALS(IGSTRT+     3)= F * DADS                                 
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=E * DADC                                 
         FUVALS(IHSTRT+     4)=E * DADS                                 
         FUVALS(IHSTRT+     3)=H * ( DEDC * DADC + E * D2ADC2 )         
         FUVALS(IHSTRT+     5)=H * ( DEDS * DADC + E * D2ADCS )         
         FUVALS(IHSTRT+     6)=H * ( DEDS * DADS + E * D2ADS2 )         
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
