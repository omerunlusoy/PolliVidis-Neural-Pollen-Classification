import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";
import Add from "./Add";
import SampleImagePreviewCard from "./SampleImagePreviewCard";
import React from "react";
import ImageCard from "./ImageCard";

const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(10),
  },
}));

const AboutUs = () => {
  const classes = useStyles();
  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={7} xs={10}>
                  <Container className={classes.container}>
                      <Typography style={{marginBottom:10}}variant="h3" component="p">
                          How Pollividis Works
                      </Typography>
                      <a target="_blank" href="https://www.google.com/" title="example">
                          <Typography style={{marginBottom:10}} variant="h6" color="textSecondary" component="p">
                              Click here to see user manual!
                          </Typography>
                      </a>
                      <Card className={classes.card}>
                          <CardActionArea>
                              <img style={{
                                  alignSelf: 'center',
                                  height: 400,
                                  width: 700,
                              }} src={"/procedure_pollividis.jpeg"}/>
                          </CardActionArea>
                      </Card>
                      <Typography style={{marginBottom:10}} variant="h6" color="textSecondary" component="p">
                          PolliVidis Pollen Sample Analysis can be divided into subroutines as follows;
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          1. Folder Iteration Procedure <br/>
                          2. Load Image <br/>
                          3. Transform into GrayScale <br/>
                          4. Obtain Otsu Threshold <br/>
                          5. Morphology Sequence <br/>
                          6. Label Image according to Otsu Threshold to obtain Regions <br/>
                          7. Extract Single Pollens from the Original Image using Label Coordinates <br/>
                          8. Forward to the Model (details in ML Implementation Section) <br/>
                          9. Label Original Image with Classification Information <br/> <br/>
                      </Typography>
                      <Typography style={{marginBottom:5}} variant="h5"  component="p">
                          In the first step, PolliVidis loads the image to the database to process later. <br/> <br/>

                          In the second step, the given image is loaded to the server memory as a Python array to ease the morphology sequence procedure. <br/> <br/>

                          In the third step, the loaded image is transformed into a GrayScale image for Otsu Thresholding. <br/> <br/>

                          In the fourth step, Otsu Threshold of the gray scaled image is obtained. <br/> <br/>

                          In the fifth step, a series of Morphology Sequences are applied to the thresholded image to finetune the labeled areas. PolliVidis has implemented two different but mostly equivalent Morphology Sequences Procedures, one for automatic procedure for general users, and one for manual morphology procedure for more sophisticated users. The alternative morphology procedures are opening, closing, erosion, dilation, area opening, and are closing. We have realized that multi erosion followed by molti dilation and finally area closing is well-suited for most pollen images. Thus, PolliVidis has automated this procedure and expect users to supply only one parameter, number of multi erosion and dilations to apply. Users can find the correct parameter of his/her pollen image with trial and error since the range is pretty small going 0 to 40 by 5. In the manual mode, users can apply all morphology procedures in diseried order by specifying the whole sequence. For example ‘E10-D10-AC100000’ means 10 erosion followed by 10 dilation and followed by 100000 area closing. <br/> <br/>

                          In the sixth step, PolliVidis labels the regions of the thresholded image with skimage library. It shows distinct regions in the image. <br/> <br/>

                          In the seventh step, each single pollen image is extracted from the original image according to the coordinates of labeling in the previous step. Thus, PolliVidis obtains the single pollen images of the original image. <br/> <br/>

                          In the eight step, PolliVidis forwards each pollen image to the ML model to classify it. <br/> <br/>

                          In the ninth and the last step, procedure labels the original image with the classification information supplied by the ML model. <br/> <br/>
                      </Typography>

                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>

              </Grid>
          </Grid>
      </div>
  );
};

export default AboutUs;
