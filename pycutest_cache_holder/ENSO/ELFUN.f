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
C  Problem name : ENSO      
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , X     , TPI   , V12   
      DOUBLE PRECISION V13   , V14   , TPIX  , TPIXS , TPIXC 
      DOUBLE PRECISION TPIXV1, C     , S     
      INTRINSIC SIN   , COS   , ATAN  
      IFSTAT = 0
      TPI    = 8.0 * ATAN( 1.0 )                        
      DO     3 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2
     *                                                        ), IELTYP
C
C  Element type : E8        
C
    1  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       X      = EPVALU(IPSTRT+     1)
       V12    = V1 * V1                                  
       V13    = V1 * V12                                 
       V14    = V12 * V12                                
       TPIX   = TPI * X                                  
       TPIXV1 = TPIX / V1                                
       C      = COS( TPIXV1 )                            
       S      = SIN( TPIXV1 )                            
       TPIXS  = TPIX * S                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V2 * C                                   
       ELSE
        FUVALS(IGSTRT+     1)= TPIXS * V2 / V12                         
        FUVALS(IGSTRT+     2)= C                                        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- TPI * TPI * V2 * C * X ** 2 / V14      
     *                         - 2.0 * TPIX * V2 * S / V13              
         FUVALS(IHSTRT+     2)=TPIXS / V12                              
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     3
C
C  Element type : E9        
C
    2  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       X      = EPVALU(IPSTRT+     1)
       V12    = V1 * V1                                  
       V13    = V1 * V12                                 
       V14    = V12 * V12                                
       TPIX   = TPI * X                                  
       TPIXV1 = TPIX / V1                                
       C      = COS( TPIXV1 )                            
       S      = SIN( TPIXV1 )                            
       TPIXC  = TPIX * C                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V2 * S                                   
       ELSE
        FUVALS(IGSTRT+     1)= - TPIXC * V2 / V12                       
        FUVALS(IGSTRT+     2)= S                                        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0 * TPIX * V2 * C / V13                
     *                         - TPI * TPI * V2 * S * X ** 2 / V14      
         FUVALS(IHSTRT+     2)=- TPIXC / V12                            
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
