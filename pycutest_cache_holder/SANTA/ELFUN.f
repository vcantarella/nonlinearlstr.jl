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
C  Problem name : SANTA     
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION PHI1  , PHI2  , LAM1  , LAM2  , LAMF  
      DOUBLE PRECISION PHIF  , LAMS  , S1    , S2    , SF    
      DOUBLE PRECISION C1    , C2    , CF    , S     , C     
      DOUBLE PRECISION C1C2S , C1C2C , C1S2S , S1C2S 
      INTRINSIC COS   , SIN   
      IFSTAT = 0
      DO     5 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3,    4
     *                                                        ), IELTYP
C
C  Element type : E         
C
    1  CONTINUE
       PHI1   = XVALUE(IELVAR(ILSTRT+     1))
       PHI2   = XVALUE(IELVAR(ILSTRT+     2))
       LAM1   = XVALUE(IELVAR(ILSTRT+     3))
       LAM2   = XVALUE(IELVAR(ILSTRT+     4))
       S1     =   SIN(PHI1)                              
       S2     =   SIN(PHI2)                              
       C1     =   COS(PHI1)                              
       C2     =   COS(PHI2)                              
       C      =   COS(LAM1-LAM2)                         
       S      =   SIN(LAM1-LAM2)                         
       C1C2S  =   C1 * C2 * S                            
       C1C2C  =   C1 * C2 * C                            
       C1S2S  =   C1 * S2 * S                            
       S1C2S  =   S1 * C2 * S                            
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)=   S1 * S2 + C1 * C2 * C                  
       ELSE
        FUVALS(IGSTRT+     1)=   C1 * S2 - S1 * C2 * C                  
        FUVALS(IGSTRT+     2)=   S1 * C2 - C1 * S2 * C                  
        FUVALS(IGSTRT+     3)=   - C1C2S                                
        FUVALS(IGSTRT+     4)=   C1C2S                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=  - S1 * S2 - C1 * C2 * C                
         FUVALS(IHSTRT+     2)=  C1 * C2 + S1 * S2 * C                  
         FUVALS(IHSTRT+     3)=  - S1 * S2 - C1 * C2 * C                
         FUVALS(IHSTRT+     4)=  S1C2S                                  
         FUVALS(IHSTRT+     5)=  C1S2S                                  
         FUVALS(IHSTRT+     6)=  - C1C2C                                
         FUVALS(IHSTRT+     7)=  - S1C2S                                
         FUVALS(IHSTRT+     8)=  - C1S2S                                
         FUVALS(IHSTRT+     9)=  C1C2C                                  
         FUVALS(IHSTRT+    10)=  - C1C2C                                
        END IF
       END IF
       GO TO     5
C
C  Element type : E3        
C
    2  CONTINUE
       PHI1   = XVALUE(IELVAR(ILSTRT+     1))
       PHI2   = XVALUE(IELVAR(ILSTRT+     2))
       LAM1   = XVALUE(IELVAR(ILSTRT+     3))
       LAMF   = EPVALU(IPSTRT+     1)
       S1     =   SIN(PHI1)                              
       S2     =   SIN(PHI2)                              
       C1     =   COS(PHI1)                              
       C2     =   COS(PHI2)                              
       C      =   COS(LAM1-LAMF)                         
       S      =   SIN(LAM1-LAMF)                         
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)=   S1 * S2 + C1 * C2 * C                  
       ELSE
        FUVALS(IGSTRT+     1)=   C1 * S2 - S1 * C2 * C                  
        FUVALS(IGSTRT+     2)=   S1 * C2 - C1 * S2 * C                  
        FUVALS(IGSTRT+     3)=   - C1 * C2 * S                          
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=  - S1 * S2 - C1 * C2 * C                
         FUVALS(IHSTRT+     2)=  C1 * C2 + S1 * S2 * C                  
         FUVALS(IHSTRT+     3)=  - S1 * S2 - C1 * C2 * C                
         FUVALS(IHSTRT+     4)=  S1 * C2 * S                            
         FUVALS(IHSTRT+     5)=  C1 * S2 * S                            
         FUVALS(IHSTRT+     6)=  - C1 * C2 * C                          
        END IF
       END IF
       GO TO     5
C
C  Element type : E2        
C
    3  CONTINUE
       PHI1   = XVALUE(IELVAR(ILSTRT+     1))
       LAM1   = XVALUE(IELVAR(ILSTRT+     2))
       PHIF   = EPVALU(IPSTRT+     1)
       LAMF   = EPVALU(IPSTRT+     2)
       S1     =   SIN(PHI1)                              
       SF     =   SIN(PHIF)                              
       C1     =   COS(PHI1)                              
       CF     =   COS(PHIF)                              
       C      =   COS(LAM1-LAMF)                         
       S      =   SIN(LAM1-LAMF)                         
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)=   S1 * SF + C1 * CF * C                  
       ELSE
        FUVALS(IGSTRT+     1)=   C1 * SF - S1 * CF * C                  
        FUVALS(IGSTRT+     2)=   - C1 * CF * S                          
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=  - S1 * SF - C1 * CF * C                
         FUVALS(IHSTRT+     2)=  S1 * CF * S                            
         FUVALS(IHSTRT+     3)=  - C1 * CF * C                          
        END IF
       END IF
       GO TO     5
C
C  Element type : E1        
C
    4  CONTINUE
       PHI1   = XVALUE(IELVAR(ILSTRT+     1))
       PHIF   = EPVALU(IPSTRT+     1)
       LAMF   = EPVALU(IPSTRT+     2)
       LAMS   = EPVALU(IPSTRT+     3)
       S1     =   SIN(PHI1)                              
       SF     =   SIN(PHIF)                              
       C1     =   COS(PHI1)                              
       CF     =   COS(PHIF)                              
       C      =   COS(LAMS-LAMF)                         
       S      =   SIN(LAMS-LAMF)                         
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)=   S1 * SF + C1 * CF * C                  
       ELSE
        FUVALS(IGSTRT+     1)=   C1 * SF - S1 * CF * C                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=  - S1 * SF - C1 * CF * C                
        END IF
       END IF
    5 CONTINUE
      RETURN
      END
