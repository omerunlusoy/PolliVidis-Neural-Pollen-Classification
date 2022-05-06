import {
    Card,
    CardActionArea,
    CardContent,
    CardMedia,
    Container,
    Grid,
    makeStyles,
    Paper,
    Typography
} from "@material-ui/core";
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

const PollenNormalNames = () => {
  return (
      <div align={"center"}>
          <Paper sx={{ width: 330, maxWidth: '100%' }} style={{align: "center"}}>
              <Typography style={{marginBottom:5}} variant="h5"  component="p">
                  Ambrosia Artemisiifolia: Common Ragweed<br/>
                  Alnus Glutinosa: Black Alder <br/>
                  Acer Negundo: Boxelder Maple <br/>
                  Betula Papyrifera: Paper Birch <br/>
                  Juglans Regia: Common Walnut <br/>
                  Artemisia Vulgaris: Common Mugwort <br/>
                  Populus Nigra: Black Poplar <br/>
                  Phleum Phleoides: Boehmer's Cat's-tail <br/>
                  Picea Abies: Norway Spruce <br/>
                  Juniperus Communis: Common Juniper <br/>
                  Ulmus Minor: Field Elm <br/>
                  Quercus Robur: Oak <br/>
                  Carpinus Betulus: Common Hornbeam <br/>
                  Ligustrum Robustrum: Wild Privet <br/>
                  Rumex Stenophyllus: Narrow-Leaf Dock <br/>
                  Ailanthus Altissima: Tree of Heaven <br/>
                  Thymbra Spicata: Mediterranean thyme <br/>
                  Rubia Peregrina: Common Wild Madder <br/>
                  Olea Europaea: Olive <br/>
                  Cichorium Intybus: Chicory <br/>
                  Chenopodium Album: Goosefoot <br/>
                  Borago Officinalis: Starflower <br/>
                  Acacia Dealbata: Silver Wattle <br/>
              </Typography>
          </Paper>
      </div>

  );
};

export default PollenNormalNames;
