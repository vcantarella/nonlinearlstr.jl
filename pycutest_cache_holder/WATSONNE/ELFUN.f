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
C  Problem name : WATSONNE  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , V4    , V5    
      DOUBLE PRECISION V6    , V7    , V8    , V9    , V10   
      DOUBLE PRECISION V11   , V12   , T1    , T2    , T3    
      DOUBLE PRECISION T4    , T5    , T6    , T7    , T8    
      DOUBLE PRECISION T9    , T10   , T11   , T12   , U     
      DOUBLE PRECISION TWOT1 , TWOT2 , TWOT3 , TWOT4 , TWOT5 
      DOUBLE PRECISION TWOT6 , TWOT7 , TWOT8 , TWOT9 , TWOT10
      DOUBLE PRECISION TWOT11, TWOT12
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
C  Element type : MSQ       
C
    2  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= - V1 * V1                                
       ELSE
        FUVALS(IGSTRT+     1)= - V1 - V1                                
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- 2.0                                    
        END IF
       END IF
       GO TO     3
C
C  Element type : MWSQ      
C
    1  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       V4     = XVALUE(IELVAR(ILSTRT+     4))
       V5     = XVALUE(IELVAR(ILSTRT+     5))
       V6     = XVALUE(IELVAR(ILSTRT+     6))
       V7     = XVALUE(IELVAR(ILSTRT+     7))
       V8     = XVALUE(IELVAR(ILSTRT+     8))
       V9     = XVALUE(IELVAR(ILSTRT+     9))
       V10    = XVALUE(IELVAR(ILSTRT+    10))
       V11    = XVALUE(IELVAR(ILSTRT+    11))
       V12    = XVALUE(IELVAR(ILSTRT+    12))
       T1     = EPVALU(IPSTRT+     1)
       T2     = EPVALU(IPSTRT+     2)
       T3     = EPVALU(IPSTRT+     3)
       T4     = EPVALU(IPSTRT+     4)
       T5     = EPVALU(IPSTRT+     5)
       T6     = EPVALU(IPSTRT+     6)
       T7     = EPVALU(IPSTRT+     7)
       T8     = EPVALU(IPSTRT+     8)
       T9     = EPVALU(IPSTRT+     9)
       T10    = EPVALU(IPSTRT+    10)
       T11    = EPVALU(IPSTRT+    11)
       T12    = EPVALU(IPSTRT+    12)
       U      = T1 * V1 + T2 * V2 + T3 * V3 + T4 * V4    
     *          + T5 * V5 + T6 * V6 + T7 * V7 + T8 * V8  
     *          + T9 * V9 + T10 * V10 + T11 * V11        
     *          + T12 * V12                              
       TWOT1  = T1 + T1                                  
       TWOT2  = T2 + T2                                  
       TWOT3  = T3 + T3                                  
       TWOT4  = T4 + T4                                  
       TWOT5  = T5 + T5                                  
       TWOT6  = T6 + T6                                  
       TWOT7  = T7 + T7                                  
       TWOT8  = T8 + T8                                  
       TWOT9  = T9 + T9                                  
       TWOT10 = T10 + T10                                
       TWOT11 = T11 + T11                                
       TWOT12 = T12 + T12                                
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= - U * U                                  
       ELSE
        FUVALS(IGSTRT+     1)= - TWOT1 * U                              
        FUVALS(IGSTRT+     2)= - TWOT2 * U                              
        FUVALS(IGSTRT+     3)= - TWOT3 * U                              
        FUVALS(IGSTRT+     4)= - TWOT4 * U                              
        FUVALS(IGSTRT+     5)= - TWOT5 * U                              
        FUVALS(IGSTRT+     6)= - TWOT6 * U                              
        FUVALS(IGSTRT+     7)= - TWOT7 * U                              
        FUVALS(IGSTRT+     8)= - TWOT8 * U                              
        FUVALS(IGSTRT+     9)= - TWOT9 * U                              
        FUVALS(IGSTRT+    10)= - TWOT10 * U                             
        FUVALS(IGSTRT+    11)= - TWOT11 * U                             
        FUVALS(IGSTRT+    12)= - TWOT12 * U                             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- TWOT1 * T1                             
         FUVALS(IHSTRT+     2)=- TWOT1 * T2                             
         FUVALS(IHSTRT+     4)=- TWOT1 * T3                             
         FUVALS(IHSTRT+     7)=- TWOT1 * T4                             
         FUVALS(IHSTRT+    11)=- TWOT1 * T5                             
         FUVALS(IHSTRT+    16)=- TWOT1 * T6                             
         FUVALS(IHSTRT+    22)=- TWOT1 * T7                             
         FUVALS(IHSTRT+    29)=- TWOT1 * T8                             
         FUVALS(IHSTRT+    37)=- TWOT1 * T9                             
         FUVALS(IHSTRT+    46)=- TWOT1 * T10                            
         FUVALS(IHSTRT+    56)=- TWOT1 * T11                            
         FUVALS(IHSTRT+    67)=- TWOT1 * T12                            
         FUVALS(IHSTRT+     3)=- TWOT2 * T2                             
         FUVALS(IHSTRT+     5)=- TWOT2 * T3                             
         FUVALS(IHSTRT+     8)=- TWOT2 * T4                             
         FUVALS(IHSTRT+    12)=- TWOT2 * T5                             
         FUVALS(IHSTRT+    17)=- TWOT2 * T6                             
         FUVALS(IHSTRT+    23)=- TWOT2 * T7                             
         FUVALS(IHSTRT+    30)=- TWOT2 * T8                             
         FUVALS(IHSTRT+    38)=- TWOT2 * T8                             
         FUVALS(IHSTRT+    47)=- TWOT2 * T10                            
         FUVALS(IHSTRT+    57)=- TWOT2 * T11                            
         FUVALS(IHSTRT+    68)=- TWOT2 * T12                            
         FUVALS(IHSTRT+     6)=- TWOT3 * T3                             
         FUVALS(IHSTRT+     9)=- TWOT3 * T4                             
         FUVALS(IHSTRT+    13)=- TWOT3 * T5                             
         FUVALS(IHSTRT+    18)=- TWOT3 * T6                             
         FUVALS(IHSTRT+    24)=- TWOT3 * T7                             
         FUVALS(IHSTRT+    31)=- TWOT3 * T8                             
         FUVALS(IHSTRT+    39)=- TWOT3 * T8                             
         FUVALS(IHSTRT+    48)=- TWOT3 * T10                            
         FUVALS(IHSTRT+    58)=- TWOT3 * T11                            
         FUVALS(IHSTRT+    69)=- TWOT3 * T12                            
         FUVALS(IHSTRT+    10)=- TWOT4 * T4                             
         FUVALS(IHSTRT+    14)=- TWOT4 * T5                             
         FUVALS(IHSTRT+    19)=- TWOT4 * T6                             
         FUVALS(IHSTRT+    25)=- TWOT4 * T7                             
         FUVALS(IHSTRT+    32)=- TWOT4 * T8                             
         FUVALS(IHSTRT+    40)=- TWOT4 * T8                             
         FUVALS(IHSTRT+    49)=- TWOT4 * T10                            
         FUVALS(IHSTRT+    59)=- TWOT4 * T11                            
         FUVALS(IHSTRT+    70)=- TWOT4 * T12                            
         FUVALS(IHSTRT+    15)=- TWOT5 * T5                             
         FUVALS(IHSTRT+    20)=- TWOT5 * T6                             
         FUVALS(IHSTRT+    26)=- TWOT5 * T7                             
         FUVALS(IHSTRT+    33)=- TWOT5 * T8                             
         FUVALS(IHSTRT+    41)=- TWOT5 * T8                             
         FUVALS(IHSTRT+    50)=- TWOT5 * T10                            
         FUVALS(IHSTRT+    60)=- TWOT5 * T11                            
         FUVALS(IHSTRT+    71)=- TWOT5 * T12                            
         FUVALS(IHSTRT+    21)=- TWOT6 * T6                             
         FUVALS(IHSTRT+    27)=- TWOT6 * T7                             
         FUVALS(IHSTRT+    34)=- TWOT6 * T8                             
         FUVALS(IHSTRT+    42)=- TWOT6 * T8                             
         FUVALS(IHSTRT+    51)=- TWOT6 * T10                            
         FUVALS(IHSTRT+    61)=- TWOT6 * T11                            
         FUVALS(IHSTRT+    72)=- TWOT6 * T12                            
         FUVALS(IHSTRT+    28)=- TWOT7 * T7                             
         FUVALS(IHSTRT+    35)=- TWOT7 * T8                             
         FUVALS(IHSTRT+    43)=- TWOT7 * T8                             
         FUVALS(IHSTRT+    52)=- TWOT7 * T10                            
         FUVALS(IHSTRT+    62)=- TWOT7 * T11                            
         FUVALS(IHSTRT+    73)=- TWOT7 * T12                            
         FUVALS(IHSTRT+    36)=- TWOT8 * T8                             
         FUVALS(IHSTRT+    44)=- TWOT8 * T8                             
         FUVALS(IHSTRT+    53)=- TWOT8 * T10                            
         FUVALS(IHSTRT+    63)=- TWOT8 * T11                            
         FUVALS(IHSTRT+    74)=- TWOT8 * T12                            
         FUVALS(IHSTRT+    45)=- TWOT9 * T9                             
         FUVALS(IHSTRT+    54)=- TWOT9 * T10                            
         FUVALS(IHSTRT+    64)=- TWOT9 * T11                            
         FUVALS(IHSTRT+    75)=- TWOT9 * T12                            
         FUVALS(IHSTRT+    55)=- TWOT10 * T10                           
         FUVALS(IHSTRT+    65)=- TWOT10 * T11                           
         FUVALS(IHSTRT+    76)=- TWOT10 * T12                           
         FUVALS(IHSTRT+    66)=- TWOT11 * T11                           
         FUVALS(IHSTRT+    77)=- TWOT11 * T12                           
         FUVALS(IHSTRT+    78)=- TWOT12 * T12                           
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
