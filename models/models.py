import torch.nn as nn

class CNN_Basic_28_64(nn.Module):
    def __init__(self):
        super(CNN_Basic_28_64, self).__init__()
        
        
        #N, C=1, H=28*(2**6), W=28*(2**6)
##############################################################

        self.cnnA1 = nn.Conv2d( in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnormA1 = nn.BatchNorm2d(16)
        
        #N, C=16+8*0, H=28*(2**5), W=28*(2**5)
##############################################################

        self.cnnA2 = nn.Conv2d( in_channels=16, out_channels=24, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnormA2 = nn.BatchNorm2d(24)     
        
        #N, C=16+8*1, H=28*(2**4), W=28*(2**4)
##############################################################

        self.cnnA3 = nn.Conv2d( in_channels=24, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnormA3 = nn.BatchNorm2d(32)          
        
        #N, C=16+8*2, H=28*(2**3), W=28*(2**3)
##############################################################

        self.cnnA4 = nn.Conv2d( in_channels=32, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnormA4 = nn.BatchNorm2d(40)           
        
        #N, C=16+8*3, H=28*(2**2), W=28*(2**2)
##############################################################

        self.cnnA5 = nn.Conv2d( in_channels=40, out_channels=48, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnormA5 = nn.BatchNorm2d(48)           

        
        

        #N, C=16+8*4, H=28, W=28
##############################################################

        self.cnn1 = nn.Conv2d( in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm1 = nn.BatchNorm2d(56)

        #N, C=16+8*5, H=14, W=14        
##############################################################

        self.cnn2 = nn.Conv2d(in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm2 = nn.BatchNorm2d(64)        
        
        #N, C=16+8*6, H=7, W=7        
##############################################################

        self.cnn3 = nn.Conv2d( in_channels=64, out_channels=72, kernel_size=(2,2), stride=(2,2), padding=(1,1) )           
        self.batchnorm3 = nn.BatchNorm2d(72)

        #N, C=16+8*7, H=4, W=4  
##############################################################

        self.cnn4 = nn.Conv2d(in_channels=72, out_channels=80, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm4 = nn.BatchNorm2d(80)        
        
        #N, C=16+8*8, H=2, W=2  
##############################################################

        self.cnn5 = nn.Conv2d(in_channels=80, out_channels=88, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm5 = nn.BatchNorm2d(88)               
        
        #N, C=16+8*9, H=2, W=2  
##############################################################

        self.cnn6 = nn.Conv2d(in_channels=88, out_channels=96, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm6 = nn.BatchNorm2d(96)         
        
        
        
        #N, C=16+8*10, H=1, W=1        
##############################################################  

        self.fc1 = nn.Linear(96 * 1 * 1, 32)   
        
        #N, C=32, H=1, W=1
##############################################################    

        self.fc2 = nn.Linear(32 * 1 * 1, 10)
        
        self.non_lin = nn.ReLU()
        
    def forward(self, x):
        
        x = self.non_lin( self.batchnormA1( self.cnnA1(x) ) )
        x = self.non_lin( self.batchnormA2( self.cnnA2(x) ) )
        x = self.non_lin( self.batchnormA3( self.cnnA3(x) ) )
        x = self.non_lin( self.batchnormA4( self.cnnA4(x) ) )
        x = self.non_lin( self.batchnormA5( self.cnnA5(x) ) )
        
        x = self.non_lin( self.batchnorm1( self.cnn1(x) ) )
        x = self.non_lin( self.batchnorm2( self.cnn2(x) ) )
        x = self.non_lin( self.batchnorm3( self.cnn3(x) ) )
        x = self.non_lin( self.batchnorm4( self.cnn4(x) ) )
        x = self.non_lin( self.batchnorm5( self.cnn5(x) ) )        
        x = self.non_lin( self.batchnorm6( self.cnn6(x) ) )

        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x





class CNN_Basic(nn.Module):
    def __init__(self):
        super(CNN_Basic, self).__init__()

        #N, C=1, H=28, W=28
##############################################################

        self.cnn1 = nn.Conv2d( in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm1 = nn.BatchNorm2d(16)

        #N, C=16, H=28, W=28        
##############################################################

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm2 = nn.BatchNorm2d(24)        
        
        #N, C=32, H=14, W=14        
##############################################################

        self.cnn3 = nn.Conv2d( in_channels=24, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm3 = nn.BatchNorm2d(32)

        #N, C=64, H=14, W=14        
##############################################################

        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm4 = nn.BatchNorm2d(40)        
        
        #N, C=128, H=7, W=7        
##############################################################

        self.cnn5 = nn.Conv2d( in_channels=40, out_channels=48, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm5 = nn.BatchNorm2d(48)

        #N, C=256, H=7, W=7        
##############################################################

        self.cnn6 = nn.Conv2d(in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(1,1) )           
        self.batchnorm6 = nn.BatchNorm2d(56)        
        
        #N, C=512, H=4, W=4        
##############################################################

        self.cnn7 = nn.Conv2d(in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm7 = nn.BatchNorm2d(64)        
        
        #N, C=1024, H=2, W=2        
##############################################################

        self.cnn8 = nn.Conv2d(in_channels=64, out_channels=72, kernel_size=(2,2), stride=(1,1), padding=(0,0) )           
        self.batchnorm8 = nn.BatchNorm2d(72)        
        
        #N, C=1024, H=1, W=1        
##############################################################  

        self.fc1 = nn.Linear(72 * 1 * 1, 36)   
        
        #N, C=128, H=1, W=1
##############################################################    

        self.fc2 = nn.Linear(36 * 1 * 1, 10)
        
        self.non_lin = nn.ReLU()
        
    def forward(self, x):
        
        # x = x.half()
        
        x = self.non_lin( self.batchnorm1( self.cnn1(x) ) )
        x = self.non_lin( self.batchnorm2( self.cnn2(x) ) )
        x = self.non_lin( self.batchnorm3( self.cnn3(x) ) )
        x = self.non_lin( self.batchnorm4( self.cnn4(x) ) )
        x = self.non_lin( self.batchnorm5( self.cnn5(x) ) )
        
        # x = x.float()
        
        x = self.non_lin( self.batchnorm6( self.cnn6(x) ) )
        x = self.non_lin( self.batchnorm7( self.cnn7(x) ) )
        x = self.non_lin( self.batchnorm8( self.cnn8(x) ) )
        
        x = x.view( -1, 72 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

def get_model( input_expand_ratio ):
    
    if( input_expand_ratio == 64):    
        return CNN_Basic_28_64( )
    elif( input_expand_ratio == 1):    
        return CNN_Basic( )
    else:
        print( 'Not Valid Input Expand Ratio ' + str(input_expand_ratio) )
        assert(0)


