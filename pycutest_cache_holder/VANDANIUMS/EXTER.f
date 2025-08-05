      DOUBLE PRECISION FUNCTION b( x0, k, h, x )                        
C                                                                       
C  compute the value of the k-th B-Spline at x                          
C                                                                       
      DOUBLE PRECISION x0, h, k, x                                      
      DOUBLE PRECISION xs, twoh                                         
      xx = x - k * h                                                    
      twoh = h + h                                                      
      IF ( xx .LE. x0 - twoh .OR. xx .GE. x0 + twoh ) THEN              
        b = 0.0D0                                                       
      ELSE IF ( xx .LE. x0 - h ) THEN                                   
        b = ( twoh + ( xx - x0 ) ** 3 ) / 6.0D0                         
      ELSE IF ( xx .GE. x0 + h ) THEN                                   
        b = ( twoh - ( xx - x0 ) ** 3 ) / 6.0D0                         
      ELSE IF ( xx .LE. x0 ) THEN                                       
        b = twoh * h * h / 3.0D0                                        
     *      - 5.0D-1 * ( twoh + xx - x0 ) * ( xx - x0 ) ** 2            
      ELSE                                                              
        b = twoh * h * h / 3.0D0                                        
     *      - 5.0D-1 * ( twoh - xx + x0 ) * ( xx - x0 ) ** 2            
      END IF                                                            
      RETURN                                                            
      END                                                               
