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
C  Problem name : PFIT3     
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION AA    , RR    , XX    , Y     , ARX   
      DOUBLE PRECISION LOGY  , A1    , B     , BA    , BX    
      DOUBLE PRECISION BAA   , BAX   , BXX   , C     , CC    
      DOUBLE PRECISION CCC   , D     , DA    , DR    , DX    
      DOUBLE PRECISION DAA   , DAR   , DAX   , DRX   , DXX   
      INTRINSIC LOG   
      IFSTAT = 0
      DO     6 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3,    4,    5
     *                                                        ), IELTYP
C
C  Element type : T1        
C
    1  CONTINUE
       AA     = XVALUE(IELVAR(ILSTRT+     1))
       RR     = XVALUE(IELVAR(ILSTRT+     2))
       XX     = XVALUE(IELVAR(ILSTRT+     3))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= AA * RR * XX                             
       ELSE
        FUVALS(IGSTRT+     1)= RR * XX                                  
        FUVALS(IGSTRT+     2)= AA * XX                                  
        FUVALS(IGSTRT+     3)= AA * RR                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=XX                                       
         FUVALS(IHSTRT+     4)=RR                                       
         FUVALS(IHSTRT+     5)=AA                                       
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
       GO TO     6
C
C  Element type : T2        
C
    2  CONTINUE
       AA     = XVALUE(IELVAR(ILSTRT+     1))
       RR     = XVALUE(IELVAR(ILSTRT+     2))
       XX     = XVALUE(IELVAR(ILSTRT+     3))
       A1     = AA + 1.0                                 
       Y      = 1.0 + XX                                 
       LOGY   = LOG( Y )                                 
       C      = Y ** ( - A1 )                            
       CC     = C / Y                                    
       CCC    = CC / Y                                   
       B      = 1.0 - C                                  
       BA     = LOGY * C                                 
       BX     = A1 * CC                                  
       BAA    = - LOGY * LOGY * C                        
       BAX    = - LOGY * BX + CC                         
       BXX    = - A1 * ( A1 + 1.0 ) * CCC                
       ARX    = AA * RR * XX                             
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= ARX * B                                  
       ELSE
        FUVALS(IGSTRT+     1)= RR * XX * B + ARX * BA                   
        FUVALS(IGSTRT+     2)= AA * XX * B                              
        FUVALS(IGSTRT+     3)= AA * RR * B + ARX * BX                   
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0 * RR * XX * BA + ARX * BAA           
         FUVALS(IHSTRT+     2)=XX * B + AA * XX * BA                    
         FUVALS(IHSTRT+     4)=RR * B + RR * XX * BX + AA * RR * BA     
     *                          + ARX * BAX                             
         FUVALS(IHSTRT+     5)=AA * B + AA * XX * BX                    
         FUVALS(IHSTRT+     6)=2.0 * AA * RR * BX + ARX * BXX           
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     6
C
C  Element type : T3        
C
    3  CONTINUE
       AA     = XVALUE(IELVAR(ILSTRT+     1))
       RR     = XVALUE(IELVAR(ILSTRT+     2))
       XX     = XVALUE(IELVAR(ILSTRT+     3))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= AA * ( AA + 1.0 ) * RR * XX * XX         
       ELSE
        FUVALS(IGSTRT+     1)= ( 2.0 *  AA + 1.0 ) * RR * XX * XX       
        FUVALS(IGSTRT+     2)= AA * ( AA + 1.0 ) * XX * XX              
        FUVALS(IGSTRT+     3)= 2.0 * AA * ( AA + 1.0 ) * RR * XX        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0 * RR * XX * XX                       
         FUVALS(IHSTRT+     2)=( 2.0 *  AA + 1.0 ) * XX * XX            
         FUVALS(IHSTRT+     4)=2.0 * ( 2.0 *  AA + 1.0 ) * RR * XX      
         FUVALS(IHSTRT+     5)=2.0 * AA * ( AA + 1.0 ) * XX             
         FUVALS(IHSTRT+     6)=2.0 * AA * ( AA + 1.0 ) * RR             
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     6
C
C  Element type : T4        
C
    4  CONTINUE
       AA     = XVALUE(IELVAR(ILSTRT+     1))
       RR     = XVALUE(IELVAR(ILSTRT+     2))
       XX     = XVALUE(IELVAR(ILSTRT+     3))
       Y      = 1.0 + XX                                 
       LOGY   = LOG( Y )                                 
       C      = Y ** ( - AA )                            
       CC     = C / Y                                    
       CCC    = CC / Y                                   
       B      = 1.0 - C                                  
       BA     = LOGY * C                                 
       BX     = AA * CC                                  
       BAA    = - LOGY * LOGY * C                        
       BAX    = - LOGY * BX + CC                         
       BXX    = - AA * ( AA + 1.0 ) * CCC                
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= RR * B                                   
       ELSE
        FUVALS(IGSTRT+     1)= RR * BA                                  
        FUVALS(IGSTRT+     2)= B                                        
        FUVALS(IGSTRT+     3)= RR * BX                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=RR * BAA                                 
         FUVALS(IHSTRT+     2)=BA                                       
         FUVALS(IHSTRT+     4)=RR * BAX                                 
         FUVALS(IHSTRT+     5)=BX                                       
         FUVALS(IHSTRT+     6)=RR * BXX                                 
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     6
C
C  Element type : T5        
C
    5  CONTINUE
       AA     = XVALUE(IELVAR(ILSTRT+     1))
       RR     = XVALUE(IELVAR(ILSTRT+     2))
       XX     = XVALUE(IELVAR(ILSTRT+     3))
       A1     = AA + 2.0                                 
       Y      = 1.0 + XX                                 
       LOGY   = LOG( Y )                                 
       C      = Y ** ( - A1 )                            
       CC     = C / Y                                    
       CCC    = CC / Y                                   
       B      = 1.0 - C                                  
       BA     = LOGY * C                                 
       BX     = A1 * CC                                  
       BAA    = - LOGY * LOGY * C                        
       BAX    = - LOGY * BX + CC                         
       BXX    = - A1 * ( A1 + 1.0 ) * CCC                
       D      = AA * ( AA + 1.0 ) * RR * XX * XX         
       DA     = ( 2.0 *  AA + 1.0 ) * RR * XX * XX       
       DR     = AA * ( AA + 1.0 ) * XX * XX              
       DX     = 2.0 * AA * ( AA + 1.0 ) * RR * XX        
       DAA    = 2.0 * RR * XX * XX                       
       DAR    = ( 2.0 *  AA + 1.0 ) * XX * XX            
       DAX    = 2.0 * ( 2.0 *  AA + 1.0 ) * RR * XX      
       DRX    = 2.0 * AA * ( AA + 1.0 ) * XX             
       DXX    = 2.0 * AA * ( AA + 1.0 ) * RR             
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= D * B                                    
       ELSE
        FUVALS(IGSTRT+     1)= DA * B + D * BA                          
        FUVALS(IGSTRT+     2)= DR * B                                   
        FUVALS(IGSTRT+     3)= DX * B + D * BX                          
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=DAA * B + 2.0 * DA * BA + D * BAA        
         FUVALS(IHSTRT+     2)=DAR * B + DR * BA                        
         FUVALS(IHSTRT+     4)=DAX * B + DA * BX + DX * BA + D * BAX    
         FUVALS(IHSTRT+     5)=DRX * B + DR * BX                        
         FUVALS(IHSTRT+     6)=DXX * B + 2.0 * DX * BX + D * BXX        
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    6 CONTINUE
      RETURN
      END
