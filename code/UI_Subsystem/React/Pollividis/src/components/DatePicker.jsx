import React, { useState } from "react";
import {Box, Card, CardActionArea, CardContent, Typography} from "@material-ui/core";

export default function DatePicker() {

    return(
        <Card>
            <CardActionArea>
                <CardContent>
                    <Typography align={"center"}  variant="h5" >
                        Select Time Interval
                    </Typography>
                    <div>
                        <Box>
                            <div>

                            </div>
                        </Box>
                    </div>
                </CardContent>
            </CardActionArea>
        </Card>

    )


}
