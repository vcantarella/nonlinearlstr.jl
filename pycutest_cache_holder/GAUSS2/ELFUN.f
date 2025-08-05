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
C  Problem name : GAUSS2    
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , X     , E     
      DOUBLE PRECISION V1E   , V2MX  , V2MX2 , TV2MX , TV2MX2
      DOUBLE PRECISION TV2MXV, TV2MXW, R     , A     , V32   
      DOUBLE PRECISION V33   , TV1E  
      INTRINSIC EXP   
      IFSTAT = 0
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
C  Element type : E2        
C
    1  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       X      = EPVALU(IPSTRT+     1)
       E      = EXP( - V2 * X )                          
       V1E    = V1 * E                                   
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1E                                      
       ELSE
        FUVALS(IGSTRT+     1)= E                                        
        FUVALS(IGSTRT+     2)= - V1E * X                                
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=- X * E                                  
         FUVALS(IHSTRT+     3)=V1E * X ** 2                             
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
       GO TO     3
C
C  Element type : E17       
C
    2  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       X      = EPVALU(IPSTRT+     1)
       V2MX   = V2 - X                                   
       V2MX2  = V2MX *V2MX                               
       TV2MX  = 2.0 * V2MX                               
       TV2MX2 = 2.0 * V2MX2                              
       R      = V2MX / V3                                
       A      = - R * R                                  
       E      = EXP( A )                                 
       V32    = V3 * V3                                  
       V33    = V3 * V32                                 
       V1E    = V1 * E                                   
       TV1E   = 2.0 * V1E                                
       TV2MXV = TV2MX / V32                              
       TV2MXW = TV2MX2 / V32                             
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1E                                      
       ELSE
        FUVALS(IGSTRT+     1)= E                                        
        FUVALS(IGSTRT+     2)= - V1E * TV2MXV                           
        FUVALS(IGSTRT+     3)= TV1E * V2MX2 / V33                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=- E * TV2MXV                             
         FUVALS(IHSTRT+     4)=E * TV2MX2 / V33                         
         FUVALS(IHSTRT+     3)=TV1E * ( TV2MXW - 1.0 ) / V32            
         FUVALS(IHSTRT+     5)=TV1E * TV2MX *                           
     *                          ( 1.0 - V2MX2 / V32 ) / V33             
         FUVALS(IHSTRT+     6)=TV1E * V2MX2 *                           
     *                           ( TV2MXW - 3.0 ) / V3 ** 4             
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
